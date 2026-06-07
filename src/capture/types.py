"""Type definitions for SPOT data capture."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LidarFrame:
    """A single LiDAR point cloud frame from Velodyne-16."""

    timestamp: float  # Unix timestamp (seconds)
    points: np.ndarray  # Nx3 float32 XYZ coordinates (meters)
    intensity: np.ndarray  # Nx1 uint8 intensity values
    frame_id: int  # Sequential frame number

    # Pose in world frame (optional, from odometry/SPOT vision)
    pose_position: Optional[np.ndarray] = None  # 3 floats: [x, y, z]
    pose_quaternion: Optional[np.ndarray] = None  # 4 floats: [x, y, z, w]


@dataclass
class CameraFrame:
    """A single camera image from SPOT body cameras."""

    timestamp: float  # Unix timestamp (seconds)
    source_name: str  # Camera identifier (e.g., "back_fisheye", "frontleft_fisheye")
    image_data: np.ndarray  # HxWx3 uint8 BGR image
    frame_id: int  # Sequential frame number for this camera

    # Camera intrinsics (optional, set during session)
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    distortion: Optional[dict] = None  # e.g., {"k1": 0.1, "k2": -0.05, ...}

    # Pose in world frame (optional, from SPOT frame tree)
    pose_position: Optional[np.ndarray] = None  # 3 floats: [x, y, z]
    pose_quaternion: Optional[np.ndarray] = None  # 4 floats: [x, y, z, w]


@dataclass
class RobotPose:
    """Robot body pose at a given time."""

    timestamp: float  # Unix timestamp (seconds)
    position: np.ndarray  # 3 floats: [x, y, z] in vision/odom frame
    quaternion: np.ndarray  # 4 floats: [x, y, z, w] (scalar-last convention)
    frame_id: str = "vision"  # Frame name (e.g., "vision", "odom")


@dataclass
class ImuData:
    """Inertial measurement unit data from SPOT."""

    timestamp: float  # Unix timestamp (seconds)
    linear_acceleration: np.ndarray  # 3 floats: [ax, ay, az] (m/s^2)
    angular_velocity: np.ndarray  # 3 floats: [wx, wy, wz] (rad/s)
    linear_acceleration_uncertainty: float = 0.0
    angular_velocity_uncertainty: float = 0.0
