"""Type definitions for recording sessions."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SessionMetadata:
    """Metadata for a recording session."""

    session_id: str  # YYYYMMDD_HHMMSS format
    start_time: datetime
    end_time: Optional[datetime] = None
    robot_hostname: str = ""
    robot_username: str = ""

    # Counters
    lidar_frame_count: int = 0
    image_frame_count: int = 0
    pose_frame_count: int = 0
    imu_frame_count: int = 0

    # Metadata
    notes: str = ""
    reconstruction_completed: bool = False
    reconstruction_timestamp: Optional[datetime] = None


@dataclass
class FrameMetadata:
    """Metadata for a single frame (LiDAR or camera)."""

    timestamp: float  # Unix timestamp
    frame_id: int  # Sequential number
    source_name: str = ""  # "velodyne" or "back_fisheye", etc.
    frame_type: str = "lidar"  # "lidar" or "camera"

    # Optional pose (SPOT vision-frame)
    pose_x: Optional[float] = None
    pose_y: Optional[float] = None
    pose_z: Optional[float] = None
    pose_qx: Optional[float] = None
    pose_qy: Optional[float] = None
    pose_qz: Optional[float] = None
    pose_qw: Optional[float] = None

    # For camera frames
    file_path: Optional[str] = None  # Relative path to image


@dataclass
class RecordingSession:
    """Active or completed recording session."""

    session_path: Path
    metadata: SessionMetadata
    lidar_count: int = 0
    image_count: int = 0
    pose_data: list = field(default_factory=list)  # List of RobotPose
    intrinsics: dict = field(default_factory=dict)  # Per-camera intrinsics

    def get_session_folder(self) -> Path:
        """Get the root session folder."""
        return self.session_path

    def get_lidar_folder(self) -> Path:
        """Get or create lidar subfolder."""
        folder = self.session_path / "lidar"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_image_folder(self) -> Path:
        """Get or create images subfolder."""
        folder = self.session_path / "images"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_reconstruction_folder(self) -> Path:
        """Get or create reconstruction output subfolder."""
        folder = self.session_path / "reconstruction"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
