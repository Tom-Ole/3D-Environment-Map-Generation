"""Type definitions for reconstruction pipeline."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OdometryResult:
    """Result of odometry computation."""

    poses: np.ndarray  # Nx7: [t, x, y, z, qx, qy, qz, qw]
    scan_indices: np.ndarray  # N indices mapping to input scans
    frame_ids: np.ndarray  # N frame IDs
    timestamps: np.ndarray  # N timestamps


@dataclass
class LoopClosureCandidate:
    """A candidate pair of poses for loop closure."""

    source_idx: int  # Index in odometry pose array
    target_idx: int  # Index in odometry pose array
    distance: float  # Spatial distance between poses (meters)
    confidence: float  # Registration confidence (0-1)


@dataclass
class LoopClosureResult:
    """Result of loop closure detection and registration."""

    candidates: list  # List of LoopClosureCandidate
    registered_pairs: dict  # {(src, tgt): relative_transform_7d}
    # Per-edge 6×6 information matrices derived from ICP covariance.
    # Used by the pose graph optimizer instead of a fixed scalar identity matrix.
    information_matrices: dict = None  # {(src, tgt): np.ndarray shape (6,6)}
    loop_count: int = 0

    def __post_init__(self):
        if self.information_matrices is None:
            self.information_matrices = {}


@dataclass
class GlobalOptimizationResult:
    """Result of global pose graph optimization."""

    optimized_poses: np.ndarray  # Nx7: [t, x, y, z, qx, qy, qz, qw]
    covariances: Optional[np.ndarray] = None  # Nx6x6 pose covariances
    residual: float = 0.0  # Final optimization residual


@dataclass
class MeshOutput:
    """Final mesh and point cloud outputs."""

    cloud_path: str  # Path to colored point cloud PLY
    mesh_ply_path: str  # Path to mesh in PLY format
    mesh_obj_path: str  # Path to mesh in OBJ format
    cloud_point_count: int = 0
    mesh_vertex_count: int = 0
    mesh_face_count: int = 0
    generation_time: float = 0.0  # Seconds


@dataclass
class ReconstructionProgress:
    """Progress update during reconstruction."""

    step_name: str  # e.g., "odometry", "loop_closure", "meshing"
    step_number: int  # Current step (1-indexed)
    total_steps: int  # Total number of steps
    progress_pct: float  # 0-100
    message: str  # Human-readable status message
    timestamp: float = 0.0  # Unix timestamp of update

    def __str__(self) -> str:
        """Format as human-readable string."""
        return f"[{self.step_number}/{self.total_steps}] {self.step_name} ({self.progress_pct:.1f}%) - {self.message}"
