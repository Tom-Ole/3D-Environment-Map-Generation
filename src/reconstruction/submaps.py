"""Local submap accumulation."""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def create_submaps(
    scans: List[np.ndarray],
    poses: np.ndarray,
    submap_size: int = 20,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Group scans into local submaps (fragments).

    Args:
        scans: List of Nx3 point clouds
        poses: Nx7 odometry poses
        submap_size: Number of scans per submap

    Returns:
        Tuple of (submaps, submap_poses) where each submap is a merged cloud
    """
    logger.info(f"Creating submaps (size={submap_size})")

    submaps = []
    submap_poses = []

    for i in range(0, len(scans), submap_size):
        start = i
        end = min(i + submap_size, len(scans))

        # Merge scans in this submap
        submap_scans = scans[start:end]
        merged = merge_scans(submap_scans, poses[start:end])

        submaps.append(merged)
        submap_poses.append(poses[start])  # Use pose of first scan

    logger.info(f"Created {len(submaps)} submaps")
    return submaps, submap_poses


def merge_scans(
    scans: List[np.ndarray],
    poses: np.ndarray,
) -> np.ndarray:
    """
    Merge multiple scans into a single cloud.

    Args:
        scans: List of point clouds
        poses: Nx7 poses corresponding to scans

    Returns:
        Merged Mx3 point cloud (in frame of first pose)
    """
    from utils.transforms import quaternion_to_rotation_matrix, invert_transform, compose_transforms

    if not scans:
        return np.array([]).reshape(0, 3)

    # Transform all scans to frame of first pose
    ref_pose = poses[0]
    ref_pos = ref_pose[1:4]
    ref_quat = ref_pose[4:8]

    # Get inverse of reference pose (world-to-ref)
    ref_pos_inv, ref_quat_inv = invert_transform(ref_pos, ref_quat)

    merged = []

    for i, scan in enumerate(scans):
        pose = poses[i]
        pos = pose[1:4]
        quat = pose[4:8]

        # Compose: ref_pose_inv * pose
        relative_pos, relative_quat = compose_transforms(
            ref_pos_inv, ref_quat_inv, pos, quat
        )

        # Transform scan
        R = quaternion_to_rotation_matrix(relative_quat)
        scan_relative = scan @ R.T + relative_pos[np.newaxis, :]

        merged.append(scan_relative)

    return np.vstack(merged)
