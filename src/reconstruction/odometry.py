"""KISS-ICP based odometry with SPOT pose seeding."""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def kiss_icp_odometry(
    scans: List[np.ndarray],
    initial_poses: Optional[np.ndarray] = None,
    max_distance: float = 0.1,
    icp_iterations: int = 10,
    voxel_size: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run KISS-ICP odometry on a sequence of point clouds.

    Args:
        scans: List of Nx3 point clouds (in sensor frame)
        initial_poses: Nx7 initial pose guesses [t, x, y, z, qx, qy, qz, qw]
                      (optional, helps seed the odometry)
        max_distance: Max correspondence distance (meters)
        icp_iterations: Number of ICP iterations per scan
        voxel_size: Voxel size for downsampling

    Returns:
        Tuple of:
        - Odometry poses (Nx7 [t, x, y, z, qx, qy, qz, qw])
        - Frame indices (N array)

    TODO: integrate kiss_icp package fully; placeholder implementation.
    """
    if not scans:
        return np.array([]), np.array([])

    try:
        import kiss_icp

        logger.info(f"Running KISS-ICP odometry on {len(scans)} scans")

        # Initialize KISS-ICP
        odometry = kiss_icp.KissICP()

        poses = []
        frame_indices = []

        for idx, scan in enumerate(scans):
            # Downsample scan
            scan_ds = downsample_cloud(scan, voxel_size)

            # Run odometry
            pose = odometry.update(scan_ds)

            # Extract pose as [x, y, z, qx, qy, qz, qw]
            x, y = pose.translation
            quat = pose.rotation.as_quat()  # [x, y, z, w]

            # Convert to our format [t, x, y, z, qx, qy, qz, qw]
            # Note: t is placeholder (use scan timestamp later)
            pose_vec = np.array([0.0, x, y, 0.0, quat[0], quat[1], quat[2], quat[3]])
            poses.append(pose_vec)
            frame_indices.append(idx)

            if (idx + 1) % 10 == 0:
                logger.debug(f"Processed {idx + 1}/{len(scans)} scans")

        poses = np.array(poses)
        frame_indices = np.array(frame_indices)

        # Seed with SPOT poses if provided
        if initial_poses is not None:
            logger.info("Seeding odometry with SPOT vision-frame poses")
            poses = seed_with_reference_poses(poses, initial_poses)

        logger.info(f"KISS-ICP odometry complete: {len(poses)} poses")
        return poses, frame_indices

    except ImportError:
        logger.warning("kiss_icp not installed, falling back to placeholder")
        return placeholder_odometry(scans, initial_poses)


def seed_with_reference_poses(
    odometry_poses: np.ndarray, reference_poses: np.ndarray
) -> np.ndarray:
    """
    Refine odometry poses using reference poses (e.g., SPOT vision frame).

    Uses Umeyama's method to align trajectories.

    Args:
        odometry_poses: Nx7 odometry poses
        reference_poses: Mx7 reference poses (typically sparser)

    Returns:
        Aligned odometry poses (Nx7)
    """
    from utils.transforms import align_odometry_to_reference

    try:
        aligned_poses, error = align_odometry_to_reference(
            odometry_poses, reference_poses
        )
        logger.info(f"Alignment error: {error:.4f} meters")
        return aligned_poses
    except Exception as e:
        logger.warning(f"Failed to align with reference poses: {e}")
        return odometry_poses


def downsample_cloud(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Downsample a point cloud using voxel grid.

    Args:
        cloud: Nx3 point cloud
        voxel_size: Size of voxel (meters)

    Returns:
        Downsampled point cloud
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd_ds = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd_ds.points)

    except Exception as e:
        logger.warning(f"Failed to downsample with Open3D: {e}")
        # Fallback: simple random sampling
        n_target = max(1, int(len(cloud) * (voxel_size / 0.05) ** 3))
        indices = np.random.choice(len(cloud), size=min(n_target, len(cloud)), replace=False)
        return cloud[indices]


def placeholder_odometry(
    scans: List[np.ndarray], initial_poses: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder odometry using SPOT poses or constant velocity model.

    TODO: remove once kiss_icp is properly integrated.
    """
    logger.warning("Using placeholder odometry (no motion estimation)")

    if initial_poses is not None:
        logger.info("Using SPOT vision-frame poses as odometry")
        return initial_poses, np.arange(len(initial_poses))

    # Fallback: assume stationary robot
    poses = []
    for i in range(len(scans)):
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Identity
        poses.append(pose)

    return np.array(poses), np.arange(len(poses))
