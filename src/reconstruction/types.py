"""Data types for LiDAR-SLAM reconstruction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class ReconstructionProgress:
    step_name: str
    step_number: int
    total_steps: int
    progress_pct: float  # 0–100, overall across all steps
    message: str


@dataclass
class ScanFrame:
    frame_id: int
    path: Path
    timestamp: float


@dataclass
class LoopEdge:
    source_idx: int          # keyframe index
    target_idx: int          # keyframe index
    T_source_to_target: np.ndarray   # 4x4 SE(3): transforms source pts into target frame
    fitness: float           # ICP overlap ratio [0, 1]
    information: np.ndarray = field(default_factory=lambda: np.eye(6) * 100.0)


@dataclass
class ReconstructionResult:
    success: bool
    cloud_path: Optional[Path] = None
    mesh_ply_path: Optional[Path] = None
    mesh_obj_path: Optional[Path] = None
    n_frames: int = 0
    n_keyframes: int = 0
    n_loop_closures: int = 0
    error_message: str = ""
