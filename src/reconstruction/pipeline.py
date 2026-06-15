"""Main offline reconstruction pipeline orchestrator."""

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from reconstruction.colorization import colorize_by_height, colorize_mesh
from reconstruction.fusion import fuse_and_downsample
from reconstruction.global_opt import build_pose_graph, optimize_pose_graph
from reconstruction.loop_closure import process_loop_closures, detect_loop_closures
from reconstruction.meshing import generate_mesh, save_colored_cloud, save_mesh
from reconstruction.odometry import kiss_icp_odometry, apply_body_to_lidar_extrinsic
from reconstruction.submaps import create_submaps
from reconstruction.types import ReconstructionProgress
from recording.session import (
    load_poses, load_session,
    load_lidar_timestamps, load_extrinsics,
    list_lidar_scans,
)
from utils.timestamps import interpolate_pose_to_timestamp

logger = logging.getLogger(__name__)


class ReconstructionPipeline:
    """
    Offline 3D reconstruction pipeline for SPOT LiDAR data.

    Stages:
      1. Load session — scans, SPOT poses, scan timestamps, body→lidar extrinsic
      2. Timestamp match — assign the correct SPOT body pose to each scan
      3. Apply extrinsic — convert body poses to LiDAR-frame poses (T_world←lidar)
      4. KISS-ICP odometry — warm-started from SPOT, with divergence reset
      5. Loop closure — spatial detection + FPFH/P2Plane ICP + info matrices
      6. Pose graph optimisation — Levenberg-Marquardt with fixed edge direction
      7. Fusion — re-project all scans to world frame at optimised poses
      8. Meshing — Poisson with statistically cleaned normals oriented to sensor centroid
      9. Export — PLY cloud + PLY/OBJ mesh
    """

    def __init__(
        self,
        session_path: Path,
        voxel_size: float = 0.05,
        loop_closure_threshold: float = 2.0,
        max_correspondence_distance: float = 0.1,
        icp_iterations: int = 50,
        progress_callback: Optional[Callable] = None,
    ):
        self.session_path = Path(session_path)
        self.voxel_size = voxel_size
        self.loop_closure_threshold = loop_closure_threshold
        self.max_correspondence_distance = max_correspondence_distance
        self.icp_iterations = icp_iterations
        self.progress_callback = progress_callback

        # Set during _load_session
        self.scans: List[np.ndarray] = []
        self.scan_timestamps: List[float] = []    # Unix timestamp per scan
        self._timestamps_are_real: bool = False   # True only when sidecar JSON files exist
        self.scan_poses: Optional[np.ndarray] = None  # Nx7 T_world←lidar per scan
        self.scan_origins: Optional[np.ndarray] = None  # Nx3 world-frame LiDAR positions
        self.body_to_lidar: Optional[np.ndarray] = None  # 4×4 extrinsic

        # Set during pipeline stages
        self.odometry_poses: Optional[np.ndarray] = None  # Nx7
        self.optimized_poses: Optional[np.ndarray] = None  # Nx7
        self.fused_cloud: Optional[np.ndarray] = None
        self.mesh = None
        self.loop_closures = None

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> bool:
        """Execute the full reconstruction pipeline."""
        try:
            start_time = time.time()
            steps = [
                self._load_session,
                self._run_odometry,
                self._run_loop_closure,
                self._run_global_optimization,
                self._run_fusion,
                self._run_meshing,
                self._export_results,
            ]
            for step in steps:
                if not step():
                    return False
            logger.info(f"Pipeline complete in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False

    # ── Progress helper ───────────────────────────────────────────────────────

    def _emit_progress(
        self, step_name: str, step_number: int, total_steps: int, pct: float, message: str
    ) -> None:
        if self.progress_callback:
            progress = ReconstructionProgress(
                step_name=step_name,
                step_number=step_number,
                total_steps=total_steps,
                progress_pct=pct,
                message=message,
                timestamp=time.time(),
            )
            try:
                self.progress_callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    # ── Stage 1: Load session ─────────────────────────────────────────────────

    def _load_session(self) -> bool:
        self._emit_progress("load_session", 1, 7, 0.0, "Loading session…")
        try:
            session = load_session(self.session_path)
            if not session:
                logger.error(f"Failed to load session from {self.session_path}")
                return False

            # ── LiDAR scans ───────────────────────────────────────────────────
            scan_files = list_lidar_scans(self.session_path)
            if not scan_files:
                logger.error("No LiDAR scans found in session")
                return False

            self.scans = []
            for scan_file in scan_files:
                pts = self._load_point_cloud(str(scan_file))
                if pts is not None:
                    self.scans.append(pts)

            if not self.scans:
                logger.error("No valid LiDAR scans loaded")
                return False

            # ── Scan timestamps ───────────────────────────────────────────────
            ts_map = load_lidar_timestamps(self.session_path)
            if ts_map:
                self.scan_timestamps = [
                    ts_map.get(i, 0.0) for i in range(len(self.scans))
                ]
                self._timestamps_are_real = True
            else:
                # Older sessions without sidecar files.  Synthesised timestamps
                # (0, 0.1, …) cannot be matched against real Unix-time SPOT poses,
                # so pose-to-scan matching is skipped for such sessions and
                # KISS-ICP runs without per-frame seeding — same behaviour as
                # the original pipeline, while still benefiting from the loop
                # closure and pose-graph fixes.
                logger.warning(
                    "No per-scan timestamp files found (old session). "
                    "Per-frame SPOT seeding will be skipped — "
                    "re-record the session to enable it."
                )
                self.scan_timestamps = [i * 0.1 for i in range(len(self.scans))]
                self._timestamps_are_real = False

            # ── Body→lidar extrinsic ──────────────────────────────────────────
            ext = load_extrinsics(self.session_path)
            bl = ext.get("body_to_lidar")
            if bl:
                from reconstruction.global_opt import transform_7d_to_4x4
                pose_7d = np.array([
                    bl["x"], bl["y"], bl["z"],
                    bl["qx"], bl["qy"], bl["qz"], bl["qw"],
                ])
                self.body_to_lidar = transform_7d_to_4x4(pose_7d)
                logger.info(f"Loaded body→lidar extrinsic: t={pose_7d[:3]}")
            else:
                logger.warning(
                    "No body→lidar extrinsic found — treating LiDAR as coincident "
                    "with robot body frame.  Points will be misaligned by ~0.3–0.5 m."
                )
                self.body_to_lidar = np.eye(4)

            # ── SPOT poses → per-scan LiDAR poses ────────────────────────────
            spot_poses = load_poses(self.session_path)
            if not self._timestamps_are_real:
                # Old session: synthesised timestamps cannot be reliably matched
                # to real SPOT pose timestamps.  Skip seeding to avoid the
                # collapse that would occur if all scans are mapped to the same
                # SPOT pose (which is what argmin gives when none of the
                # synthesised values fall inside the real timestamp range).
                self.scan_poses = None
            elif spot_poses is not None and len(spot_poses) > 0:
                self.scan_poses = self._match_poses_to_scans(spot_poses)
            else:
                logger.warning(
                    "No SPOT poses found — KISS-ICP will start from origin."
                )
                self.scan_poses = None

            logger.info(
                f"Loaded {len(self.scans)} scans, "
                f"{len(spot_poses) if spot_poses is not None else 0} SPOT poses"
            )
            self._emit_progress(
                "load_session", 1, 7, 100.0, f"Loaded {len(self.scans)} scans"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}", exc_info=True)
            return False

    def _match_poses_to_scans(self, spot_poses: np.ndarray) -> np.ndarray:
        """
        Interpolate SPOT poses to each scan's timestamp, then apply the
        body→lidar extrinsic to obtain T_world←lidar per scan.

        spot_poses: Nx8 [timestamp, x, y, z, qx, qy, qz, qw]
        Returns:    Mx7 [x, y, z, qx, qy, qz, qw]  (M = number of scans)
        """
        pose_ts = spot_poses[:, 0]
        positions = spot_poses[:, 1:4]
        quaternions = spot_poses[:, 4:8]

        body_poses = []
        for t in self.scan_timestamps:
            pos, quat = interpolate_pose_to_timestamp(t, pose_ts, positions, quaternions)
            if pos is None:
                # Out of range: use nearest pose
                idx = int(np.argmin(np.abs(pose_ts - t)))
                pos = positions[idx].copy()
                quat = quaternions[idx].copy()
            body_poses.append(np.concatenate([pos, quat]))

        body_poses_arr = np.array(body_poses)   # Mx7 body poses

        # Apply body→lidar extrinsic: T_world←lidar = T_world←body @ T_body←lidar
        lidar_poses = apply_body_to_lidar_extrinsic(body_poses_arr, self.body_to_lidar)
        return lidar_poses   # Mx7

    # ── Stage 2: Odometry ─────────────────────────────────────────────────────

    def _run_odometry(self) -> bool:
        self._emit_progress("odometry", 2, 7, 0.0, "Running KISS-ICP odometry…")
        try:
            self.odometry_poses, _ = kiss_icp_odometry(
                self.scans,
                scan_poses=self.scan_poses,
                max_distance=self.max_correspondence_distance,
                icp_iterations=self.icp_iterations,
                voxel_size=self.voxel_size,
            )
            self._emit_progress(
                "odometry", 2, 7, 100.0,
                f"Computed {len(self.odometry_poses)} poses"
            )
            return True
        except Exception as e:
            logger.error(f"Odometry failed: {e}", exc_info=True)
            return False

    # ── Stage 3: Loop closure ─────────────────────────────────────────────────

    def _run_loop_closure(self) -> bool:
        self._emit_progress("loop_closure", 3, 7, 0.0, "Detecting loop closures…")
        try:
            # Build Nx8 array expected by detect_loop_closures (prepend zero timestamp)
            poses_8d = np.hstack([
                np.zeros((len(self.odometry_poses), 1)),
                self.odometry_poses,
            ])
            candidates = detect_loop_closures(
                poses_8d,
                self.scans,
                distance_threshold=self.loop_closure_threshold,
                min_frame_gap=10,
            )

            if not candidates:
                logger.info("No loop closures detected")
                self.loop_closures = None
                self._emit_progress("loop_closure", 3, 7, 100.0, "No loop closures found")
                return True

            self.loop_closures = process_loop_closures(
                candidates,
                self.scans,
                self.odometry_poses,
                max_correspondence_distance=self.max_correspondence_distance,
                voxel_size=self.voxel_size,
            )
            self._emit_progress(
                "loop_closure", 3, 7, 100.0,
                f"Registered {self.loop_closures.loop_count} loop closures"
            )
            return True

        except Exception as e:
            logger.error(f"Loop closure failed: {e}", exc_info=True)
            self.loop_closures = None
            return True  # non-fatal: continue without loop closure

    # ── Stage 4: Global optimisation ─────────────────────────────────────────

    def _run_global_optimization(self) -> bool:
        self._emit_progress("global_opt", 4, 7, 0.0, "Building pose graph…")
        try:
            from reconstruction.types import LoopClosureResult

            lc = self.loop_closures or LoopClosureResult(
                candidates=[], registered_pairs={}, information_matrices={}
            )

            pose_graph = build_pose_graph(self.odometry_poses, lc)
            self._emit_progress("global_opt", 4, 7, 50.0, "Optimising pose graph…")

            self.optimized_poses, residual = optimize_pose_graph(
                pose_graph, max_iterations=100
            )
            self._emit_progress(
                "global_opt", 4, 7, 100.0,
                f"Optimisation residual: {residual:.4f}"
            )
            return True

        except Exception as e:
            logger.error(f"Global optimisation failed: {e}", exc_info=True)
            # Strip any leading timestamp column so fusion gets Nx7
            self.optimized_poses = (
                self.odometry_poses[:, 1:]
                if self.odometry_poses.shape[1] == 8
                else self.odometry_poses
            )
            return True  # non-fatal

    # ── Stage 5: Fusion ───────────────────────────────────────────────────────

    def _run_fusion(self) -> bool:
        self._emit_progress("fusion", 5, 7, 0.0, "Fusing point clouds…")
        try:
            self.fused_cloud = fuse_and_downsample(
                self.scans,
                self.optimized_poses,
                voxel_size=self.voxel_size,
            )

            # Extract scan origins (world-frame LiDAR positions) for normal orientation.
            self.scan_origins = self.optimized_poses[:len(self.scans), :3]

            self._emit_progress(
                "fusion", 5, 7, 100.0,
                f"Fused cloud: {len(self.fused_cloud)} points"
            )
            return True

        except Exception as e:
            logger.error(f"Fusion failed: {e}", exc_info=True)
            return False

    # ── Stage 6: Meshing ──────────────────────────────────────────────────────

    def _run_meshing(self) -> bool:
        self._emit_progress("meshing", 6, 7, 0.0, "Generating mesh…")
        try:
            colors = colorize_by_height(self.fused_cloud)

            mesh_output, self.mesh = generate_mesh(
                self.fused_cloud,
                colors=colors,
                voxel_size=self.voxel_size,
                depth=8,
                scan_origins=self.scan_origins,
            )

            self._emit_progress(
                "meshing", 6, 7, 100.0,
                f"Mesh: {mesh_output.mesh_vertex_count} vertices"
            )
            return True

        except Exception as e:
            logger.error(f"Meshing failed: {e}", exc_info=True)
            return False

    # ── Stage 7: Export ───────────────────────────────────────────────────────

    def _export_results(self) -> bool:
        self._emit_progress("export", 7, 7, 0.0, "Exporting results…")
        try:
            recon_path = self.session_path / "reconstruction"
            recon_path.mkdir(parents=True, exist_ok=True)

            if self.fused_cloud is not None:
                colors = colorize_by_height(self.fused_cloud)
                save_colored_cloud(
                    self.fused_cloud, colors,
                    str(recon_path / "cloud_optimized.ply")
                )
                self._emit_progress("export", 7, 7, 50.0, "Saved point cloud")

            if self.mesh is not None:
                save_mesh(self.mesh, str(recon_path / "mesh.ply"), format="ply")
                save_mesh(self.mesh, str(recon_path / "mesh.obj"), format="obj")
                self._emit_progress("export", 7, 7, 100.0, "Saved mesh")

            return True

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return False

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_point_cloud(path: str) -> Optional[np.ndarray]:
        """Load Nx3 float32 XYZ from a PLY file."""
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to load point cloud {path}: {e}")
            return None
