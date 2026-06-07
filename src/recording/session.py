"""Session management and persistence."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from recording.types import RecordingSession, SessionMetadata

logger = logging.getLogger(__name__)


def create_session(
    output_dir: Path,
    robot_hostname: str = "",
    robot_username: str = "",
) -> RecordingSession:
    """
    Create a new recording session with auto-timestamped folder.

    Args:
        output_dir: Base directory for recordings
        robot_hostname: Robot hostname (for metadata)
        robot_username: Robot username (for metadata)

    Returns:
        RecordingSession object
    """
    # Create timestamped folder: YYYYMMDD_HHMMSS
    now = datetime.now()
    session_id = now.strftime("%Y%m%d_%H%M%S")
    session_path = output_dir / session_id

    session_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created session folder: {session_path}")

    metadata = SessionMetadata(
        session_id=session_id,
        start_time=now,
        robot_hostname=robot_hostname,
        robot_username=robot_username,
    )

    session = RecordingSession(
        session_path=session_path,
        metadata=metadata,
    )

    return session


def save_session_metadata(session: RecordingSession) -> None:
    """
    Save session metadata to JSON file.

    Args:
        session: RecordingSession to save
    """
    metadata_path = session.session_path / "metadata.json"

    metadata_dict = {
        "session_id": session.metadata.session_id,
        "start_time": session.metadata.start_time.isoformat(),
        "end_time": session.metadata.end_time.isoformat()
        if session.metadata.end_time
        else None,
        "robot_hostname": session.metadata.robot_hostname,
        "robot_username": session.metadata.robot_username,
        "lidar_frame_count": session.metadata.lidar_frame_count,
        "image_frame_count": session.metadata.image_frame_count,
        "pose_frame_count": session.metadata.pose_frame_count,
        "imu_frame_count": session.metadata.imu_frame_count,
        "notes": session.metadata.notes,
        "reconstruction_completed": session.metadata.reconstruction_completed,
        "reconstruction_timestamp": session.metadata.reconstruction_timestamp.isoformat()
        if session.metadata.reconstruction_timestamp
        else None,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    logger.debug(f"Saved session metadata to {metadata_path}")


def load_session(session_path: Path) -> Optional[RecordingSession]:
    """
    Load an existing recording session from disk.

    Args:
        session_path: Path to session folder

    Returns:
        RecordingSession object, or None if invalid

    Raises:
        FileNotFoundError: If session folder does not exist
        json.JSONDecodeError: If metadata.json is invalid
    """
    metadata_path = session_path / "metadata.json"

    if not metadata_path.exists():
        logger.warning(f"No metadata.json found in {session_path}")
        return None

    with open(metadata_path, "r") as f:
        metadata_dict = json.load(f)

    metadata = SessionMetadata(
        session_id=metadata_dict["session_id"],
        start_time=datetime.fromisoformat(metadata_dict["start_time"]),
        end_time=datetime.fromisoformat(metadata_dict["end_time"])
        if metadata_dict.get("end_time")
        else None,
        robot_hostname=metadata_dict.get("robot_hostname", ""),
        robot_username=metadata_dict.get("robot_username", ""),
        lidar_frame_count=metadata_dict.get("lidar_frame_count", 0),
        image_frame_count=metadata_dict.get("image_frame_count", 0),
        pose_frame_count=metadata_dict.get("pose_frame_count", 0),
        imu_frame_count=metadata_dict.get("imu_frame_count", 0),
        notes=metadata_dict.get("notes", ""),
        reconstruction_completed=metadata_dict.get("reconstruction_completed", False),
        reconstruction_timestamp=datetime.fromisoformat(
            metadata_dict["reconstruction_timestamp"]
        )
        if metadata_dict.get("reconstruction_timestamp")
        else None,
    )

    session = RecordingSession(
        session_path=session_path,
        metadata=metadata,
    )

    logger.info(f"Loaded session from {session_path}")
    return session


def load_poses(session_path: Path) -> Optional[np.ndarray]:
    """
    Load poses from poses.npy file.

    Args:
        session_path: Path to session folder

    Returns:
        Nx7 array [timestamp, x, y, z, qx, qy, qz, qw], or None if not found
    """
    poses_path = session_path / "poses.npy"

    if not poses_path.exists():
        logger.warning(f"No poses.npy found in {session_path}")
        return None

    poses = np.load(poses_path)
    logger.debug(f"Loaded {len(poses)} poses from {poses_path}")
    return poses


def save_poses(session_path: Path, poses: np.ndarray) -> None:
    """
    Save poses to poses.npy file.

    Args:
        session_path: Path to session folder
        poses: Nx7 array [timestamp, x, y, z, qx, qy, qz, qw]
    """
    poses_path = session_path / "poses.npy"
    np.save(poses_path, poses)
    logger.debug(f"Saved {len(poses)} poses to {poses_path}")


def load_intrinsics(session_path: Path) -> dict:
    """
    Load camera intrinsics from intrinsics.json file.

    Args:
        session_path: Path to session folder

    Returns:
        Dict mapping camera names to intrinsic parameters
    """
    intrinsics_path = session_path / "intrinsics.json"

    if not intrinsics_path.exists():
        logger.warning(f"No intrinsics.json found in {session_path}")
        return {}

    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)

    logger.debug(f"Loaded intrinsics for {len(intrinsics)} cameras")
    return intrinsics


def save_intrinsics(session_path: Path, intrinsics: dict) -> None:
    """
    Save camera intrinsics to intrinsics.json file.

    Args:
        session_path: Path to session folder
        intrinsics: Dict mapping camera names to intrinsic parameters
    """
    intrinsics_path = session_path / "intrinsics.json"

    with open(intrinsics_path, "w") as f:
        json.dump(intrinsics, f, indent=2)

    logger.debug(f"Saved intrinsics for {len(intrinsics)} cameras")


def list_lidar_scans(session_path: Path) -> list:
    """
    List all LiDAR scans in a session.

    Returns:
        List of .ply file paths in lidar folder, sorted by frame number
    """
    lidar_path = session_path / "lidar"

    if not lidar_path.exists():
        return []

    scan_files = sorted(lidar_path.glob("*.ply"))
    logger.debug(f"Found {len(scan_files)} LiDAR scans")
    return scan_files


def list_images(session_path: Path) -> list:
    """
    List all images in a session.

    Returns:
        List of image file paths in images folder, sorted by filename
    """
    images_path = session_path / "images"

    if not images_path.exists():
        return []

    image_files = sorted(images_path.glob("*.png")) + sorted(images_path.glob("*.jpg"))
    logger.debug(f"Found {len(image_files)} images")
    return image_files
