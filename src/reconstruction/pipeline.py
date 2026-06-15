"""
LiDAR-SLAM reconstruction pipeline.

Six-stage architecture based on KISS-ICP / KISS-SLAM:

  Stage 1  load_data      – scan discovery, SPOT VIO pose load
  Stage 2  odometry       – KISS-ICP frame-to-frame registration
  Stage 3  keyframes      – greedy distance/angle keyframe selection
  Stage 4  loop_closure   – KD-tree candidate search + ICP verification
  Stage 5  pose_graph     – Open3D Levenberg-Marquardt optimisation
  Stage 6  fusion         – point cloud fusion + Poisson meshing + save

References:
  KISS-ICP:   Vizzo et al., RA-L 2023
  Pose graph: Choi et al., CVPR 2015 (via open3d)
"""

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from reconstruction.types import (
    LoopEdge,
    ReconstructionProgress,
    ScanFrame,
)
from reconstruction.io import (
    load_scan_frames,
    load_point_cloud,
    load_spot_poses,
)
from reconstruction.odometry import run_odometry
from reconstruction.loop_closure import detect_loop_closures
from reconstruction.global_opt import optimize_pose_graph
from reconstruction.fusion import fuse_point_clouds, reconstruct_mesh, save_results

logger = logging.getLogger(__name__)

_TOTAL_STEPS = 6

# Keyframe selection thresholds
_KF_MIN_DIST = 0.3     # metres – minimum translation to accept a new keyframe
_KF_MIN_ANGLE = 0.087  # radians – ~5°, minimum rotation to accept a new keyframe


class ReconstructionPipeline:
    """
    End-to-end LiDAR-SLAM reconstruction pipeline.

    Instantiated and run by ReconstructWorker (QThread) in the GUI.
    """

    def __init__(
        self,
        session_path: Path,
        voxel_size: float = 0.05,
        loop_closure_threshold: float = 2.0,
        max_correspondence_distance: float = 0.1,
        icp_iterations: int = 50,
        progress_callback: Optional[Callable[[ReconstructionProgress], None]] = None,
    ):
        """
        Args:
            session_path: Root folder of the recorded session.
            voxel_size: Final output voxel resolution (m).
            loop_closure_threshold: Spatial search radius for loop candidates (m).
            max_correspondence_distance: ICP max correspondence distance for
                loop closure verification (m).
            icp_iterations: Max ICP iterations for loop closure verification.
            progress_callback: Called with ReconstructionProgress after each
                sub-stage.  Must be thread-safe (GUI connects via Qt Signal).
        """
        self.session_path = Path(session_path)
        self.voxel_size = voxel_size
        self.loop_closure_threshold = loop_closure_threshold
        self.max_correspondence_distance = max_correspondence_distance
        self.icp_iterations = icp_iterations
        self.progress_callback = progress_callback

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Execute the full six-stage pipeline. Returns True on success."""
        t_start = time.time()
        try:
            # Stage 1 – load data
            self._emit(1, "load_data", 0.0, "Loading LiDAR scans…")
            frames = load_scan_frames(self.session_path)
            spot_poses = load_spot_poses(self.session_path)
            self._emit(1, "load_data", 100.0, f"Loaded {len(frames)} scans")

            # Stage 2 – KISS-ICP odometry
            self._emit(2, "odometry", 0.0, "Starting KISS-ICP odometry…")
            all_poses = self._run_odometry(frames)
            self._emit(2, "odometry", 100.0, f"Estimated {len(all_poses)} poses")

            # Stage 3 – keyframe selection
            self._emit(3, "keyframes", 0.0, "Selecting keyframes…")
            kf_indices = _select_keyframes(all_poses)
            kf_poses = [all_poses[i] for i in kf_indices]
            kf_paths = [frames[i].path for i in kf_indices]
            self._emit(
                3, "keyframes", 100.0,
                f"Selected {len(kf_indices)} keyframes from {len(frames)} frames"
            )

            # Stage 4 – loop closure
            self._emit(4, "loop_closure", 0.0,
                       f"Searching for loops in {len(kf_indices)} keyframes…")
            loop_edges = self._detect_loops(kf_poses, kf_paths)
            self._emit(4, "loop_closure", 100.0,
                       f"Found {len(loop_edges)} loop closures")

            # Stage 5 – pose graph optimisation
            self._emit(5, "pose_graph", 0.0, "Optimising pose graph…")
            optimized_poses = _run_opt(kf_poses, loop_edges)
            self._emit(5, "pose_graph", 100.0, "Pose graph optimised")

            # Stage 6 – fusion + meshing
            self._emit(6, "fusion", 0.0, "Fusing point clouds…")
            cloud, mesh = self._fuse_and_mesh(kf_paths, optimized_poses)

            output_dir = self.session_path / "reconstruction"
            save_results(cloud, mesh, output_dir)
            self._emit(6, "fusion", 100.0, f"Results saved to {output_dir.name}/")

            elapsed = time.time() - t_start
            logger.info(f"Reconstruction complete in {elapsed:.1f} s")
            return True

        except Exception as exc:
            logger.error(f"Pipeline failed: {exc}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_odometry(self, frames: List[ScanFrame]) -> List[np.ndarray]:
        n = len(frames)

        def _progress(done: int, total: int) -> None:
            pct = done / total * 100.0
            self._emit(2, "odometry", pct, f"Frame {done}/{total}")

        # KISS-ICP voxel size: use a coarser grid than the output voxel size.
        # 1.0 m is the KISS-ICP default and works well for typical indoor ranges.
        kiss_voxel = max(self.voxel_size * 10, 1.0)

        return run_odometry(
            frames,
            load_cloud_fn=load_point_cloud,
            voxel_size=kiss_voxel,
            progress_cb=_progress,
        )

    def _detect_loops(
        self,
        kf_poses: List[np.ndarray],
        kf_paths,
    ) -> List[LoopEdge]:
        def _progress(done: int, total: int) -> None:
            pct = done / total * 100.0
            self._emit(4, "loop_closure", pct, f"Keyframe {done}/{total}")

        # ICP correspondence distance for loop verification: use the larger of
        # the user-supplied distance and twice the output voxel size.
        icp_corr_dist = max(self.max_correspondence_distance, self.voxel_size * 2)

        return detect_loop_closures(
            kf_poses,
            kf_paths,
            voxel_size=max(self.voxel_size * 2, 0.05),
            threshold=self.loop_closure_threshold,
            max_correspondence_distance=icp_corr_dist,
            icp_iterations=self.icp_iterations,
            progress_cb=_progress,
        )

    def _fuse_and_mesh(self, kf_paths, optimized_poses):
        n = len(kf_paths)

        def _progress(done: int, total: int) -> None:
            pct = done / total * 70.0   # 0–70 % for fusion, 70–100 % for meshing
            self._emit(6, "fusion", pct, f"Fusing frame {done}/{total}")

        cloud = fuse_point_clouds(kf_paths, optimized_poses, self.voxel_size, _progress)

        self._emit(6, "fusion", 75.0, "Estimating normals & reconstructing mesh…")
        mesh = reconstruct_mesh(cloud)
        return cloud, mesh

    # ------------------------------------------------------------------
    # Progress helper
    # ------------------------------------------------------------------

    def _emit(self, step: int, name: str, stage_pct: float, msg: str) -> None:
        """Convert per-stage percentage to overall and invoke callback."""
        if self.progress_callback is None:
            return
        overall_pct = ((step - 1) + stage_pct / 100.0) / _TOTAL_STEPS * 100.0
        self.progress_callback(
            ReconstructionProgress(
                step_name=name,
                step_number=step,
                total_steps=_TOTAL_STEPS,
                progress_pct=overall_pct,
                message=msg,
            )
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _select_keyframes(poses: List[np.ndarray]) -> List[int]:
    """
    Greedy keyframe selection: accept a frame if the robot moved
    _KF_MIN_DIST metres or rotated _KF_MIN_ANGLE radians since the
    last accepted keyframe.

    Always includes the first and last frame.
    """
    if not poses:
        return []

    indices = [0]
    last_T = poses[0]

    for i in range(1, len(poses)):
        T_rel = np.linalg.inv(last_T) @ poses[i]
        dist = float(np.linalg.norm(T_rel[:3, 3]))
        angle = _rotation_angle(T_rel[:3, :3])
        if dist >= _KF_MIN_DIST or angle >= _KF_MIN_ANGLE:
            indices.append(i)
            last_T = poses[i]

    if indices[-1] != len(poses) - 1:
        indices.append(len(poses) - 1)

    return indices


def _rotation_angle(R: np.ndarray) -> float:
    """Rotation angle (radians) from a 3x3 rotation matrix."""
    trace = float(np.clip(np.trace(R), -1.0, 3.0))
    return float(np.arccos((trace - 1.0) / 2.0))


def _run_opt(
    kf_poses: List[np.ndarray],
    loop_edges: List[LoopEdge],
) -> List[np.ndarray]:
    """Run pose graph optimization; fall through to raw odometry on failure."""
    if len(kf_poses) < 2:
        return kf_poses
    try:
        return optimize_pose_graph(kf_poses, loop_edges)
    except Exception as e:
        logger.warning(f"Pose graph optimisation failed ({e}), using raw odometry poses")
        return kf_poses
