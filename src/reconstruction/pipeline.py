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
from reconstruction.odometry import kiss_icp_odometry
from reconstruction.submaps import create_submaps
from reconstruction.types import ReconstructionProgress
from recording.session import load_intrinsics, load_poses, load_session, list_images, list_lidar_scans

logger = logging.getLogger(__name__)


class ReconstructionPipeline:
    """
    Offline 3D reconstruction pipeline for SPOT LiDAR data.

    Orchestrates: odometry → loop closure → global optimization → fusion → meshing → colorization
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
        """
        Initialize reconstruction pipeline.

        Args:
            session_path: Path to recording session
            voxel_size: Voxel size for downsampling
            loop_closure_threshold: Spatial proximity threshold (meters)
            max_correspondence_distance: ICP correspondence distance
            icp_iterations: Number of ICP iterations
            progress_callback: Optional callback(ReconstructionProgress) for progress updates
        """
        self.session_path = Path(session_path)
        self.voxel_size = voxel_size
        self.loop_closure_threshold = loop_closure_threshold
        self.max_correspondence_distance = max_correspondence_distance
        self.icp_iterations = icp_iterations
        self.progress_callback = progress_callback

        self.scans: List[np.ndarray] = []
        self.scan_timestamps: List[float] = []
        self.odometry_poses: Optional[np.ndarray] = None
        self.optimized_poses: Optional[np.ndarray] = None
        self.fused_cloud: Optional[np.ndarray] = None
        self.mesh = None

    def run(self) -> bool:
        """
        Execute the full reconstruction pipeline.

        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()

            # Load session data
            if not self._load_session():
                return False

            # Odometry
            if not self._run_odometry():
                return False

            # Loop closure
            if not self._run_loop_closure():
                return False

            # Global optimization
            if not self._run_global_optimization():
                return False

            # Fusion
            if not self._run_fusion():
                return False

            # Meshing
            if not self._run_meshing():
                return False

            # Export
            if not self._export_results():
                return False

            elapsed = time.time() - start_time
            logger.info(f"Pipeline complete in {elapsed:.1f} seconds")
            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False

    def _emit_progress(
        self, step_name: str, step_number: int, total_steps: int, pct: float, message: str
    ) -> None:
        """Emit progress callback."""
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

    def _load_session(self) -> bool:
        """Load session data from disk."""
        self._emit_progress("load_session", 1, 9, 10.0, "Loading session...")

        try:
            # Load session metadata
            session = load_session(self.session_path)
            if not session:
                logger.error(f"Failed to load session from {self.session_path}")
                return False

            # Load LiDAR scans
            scan_files = list_lidar_scans(self.session_path)
            if not scan_files:
                logger.error("No LiDAR scans found in session")
                return False

            self.scans = []
            for scan_file in scan_files:
                try:
                    pcd = self._load_point_cloud(str(scan_file))
                    if pcd is not None:
                        self.scans.append(pcd)
                except Exception as e:
                    logger.warning(f"Failed to load scan {scan_file}: {e}")

            if not self.scans:
                logger.error("No valid LiDAR scans loaded")
                return False

            # Load poses
            poses = load_poses(self.session_path)
            if poses is not None:
                self.odometry_poses = poses

            logger.info(
                f"Loaded session: {len(self.scans)} scans, "
                f"{len(self.odometry_poses) if self.odometry_poses is not None else 0} poses"
            )

            self._emit_progress("load_session", 1, 9, 100.0, f"Loaded {len(self.scans)} scans")
            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    def _run_odometry(self) -> bool:
        """Run odometry estimation."""
        self._emit_progress("odometry", 2, 9, 0.0, "Running odometry...")

        try:
            initial_poses = self.odometry_poses
            self.odometry_poses, _ = kiss_icp_odometry(
                self.scans,
                initial_poses=initial_poses,
                max_distance=self.max_correspondence_distance,
                icp_iterations=self.icp_iterations,
                voxel_size=self.voxel_size,
            )

            self._emit_progress(
                "odometry", 2, 9, 100.0, f"Computed {len(self.odometry_poses)} poses"
            )
            return True

        except Exception as e:
            logger.error(f"Odometry failed: {e}")
            return False

    def _run_loop_closure(self) -> bool:
        """Detect and register loop closures."""
        self._emit_progress("loop_closure", 3, 9, 0.0, "Detecting loop closures...")

        try:
            candidates = detect_loop_closures(
                self.odometry_poses,
                self.scans,
                distance_threshold=self.loop_closure_threshold,
                min_frame_gap=10,
            )

            if not candidates:
                logger.info("No loop closures detected")
                self._emit_progress("loop_closure", 3, 9, 100.0, "No loop closures")
                return True

            loop_result = process_loop_closures(
                candidates,
                self.scans,
                self.odometry_poses,
                max_correspondence_distance=self.max_correspondence_distance,
            )

            self._emit_progress(
                "loop_closure", 3, 9, 100.0, f"Registered {loop_result.loop_count} loops"
            )

            # Store for global optimization
            self.loop_closures = loop_result
            return True

        except Exception as e:
            logger.error(f"Loop closure failed: {e}")
            self.loop_closures = None
            return False

    def _run_global_optimization(self) -> bool:
        """Build and optimize pose graph."""
        self._emit_progress("global_opt", 4, 9, 0.0, "Building pose graph...")

        try:
            # Create dummy loop closure result if none found
            from reconstruction.types import LoopClosureResult

            if not hasattr(self, "loop_closures") or self.loop_closures is None:
                self.loop_closures = LoopClosureResult(candidates=[], registered_pairs={})

            pose_graph = build_pose_graph(self.odometry_poses, self.loop_closures)

            self._emit_progress("global_opt", 4, 9, 50.0, "Optimizing pose graph...")

            self.optimized_poses, residual = optimize_pose_graph(
                pose_graph, max_iterations=100
            )

            self._emit_progress(
                "global_opt", 4, 9, 100.0, f"Optimization residual: {residual:.4f}"
            )
            return True

        except Exception as e:
            logger.error(f"Global optimization failed: {e}")
            self.optimized_poses = self.odometry_poses
            return False

    def _run_fusion(self) -> bool:
        """Fuse scans and downsample."""
        self._emit_progress("fusion", 5, 9, 0.0, "Fusing point clouds...")

        try:
            self.fused_cloud = fuse_and_downsample(
                self.scans,
                self.optimized_poses,
                voxel_size=self.voxel_size,
            )

            self._emit_progress(
                "fusion", 5, 9, 100.0, f"Fused cloud: {len(self.fused_cloud)} points"
            )
            return True

        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            return False

    def _run_meshing(self) -> bool:
        """Generate mesh from fused cloud."""
        self._emit_progress("meshing", 6, 9, 0.0, "Generating mesh...")

        try:
            colors = colorize_by_height(self.fused_cloud)

            mesh_output, mesh = generate_mesh(
                self.fused_cloud,
                colors=colors,
                voxel_size=self.voxel_size,
                depth=8,
            )

            self.mesh = mesh
            self._emit_progress(
                "meshing", 6, 9, 100.0, f"Mesh: {mesh_output.mesh_vertex_count} vertices"
            )
            return True

        except Exception as e:
            logger.error(f"Meshing failed: {e}")
            return False

    def _export_results(self) -> bool:
        """Export final results to disk."""
        self._emit_progress("export", 7, 9, 0.0, "Exporting results...")

        try:
            recon_path = self.session_path / "reconstruction"
            recon_path.mkdir(parents=True, exist_ok=True)

            # Save colored point cloud
            if self.fused_cloud is not None:
                colors = colorize_by_height(self.fused_cloud)
                cloud_path = recon_path / "cloud_optimized.ply"
                save_colored_cloud(self.fused_cloud, colors, str(cloud_path))

                self._emit_progress("export", 7, 9, 50.0, "Saved point cloud")

            # Save mesh
            if self.mesh is not None:
                mesh_ply = recon_path / "mesh.ply"
                mesh_obj = recon_path / "mesh.obj"

                save_mesh(self.mesh, str(mesh_ply), format="ply")
                save_mesh(self.mesh, str(mesh_obj), format="obj")

                self._emit_progress("export", 7, 9, 100.0, "Saved mesh")

            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    @staticmethod
    def _load_point_cloud(path: str) -> Optional[np.ndarray]:
        """Load point cloud from file."""
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to load point cloud {path}: {e}")
            return None
