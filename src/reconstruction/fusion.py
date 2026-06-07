"""Point cloud fusion and voxel downsampling."""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def fuse_and_downsample(
    scans: List[np.ndarray],
    optimized_poses: np.ndarray,
    voxel_size: float = 0.05,
) -> np.ndarray:
    """
    Transform all scans to world frame and fuse into a single cloud.

    Args:
        scans: List of Nx3 point clouds
        optimized_poses: Nx7 optimized poses [t, x, y, z, qx, qy, qz, qw]
        voxel_size: Voxel size for downsampling

    Returns:
        Fused and downsampled point cloud (Mx3)
    """
    from utils.transforms import quaternion_to_rotation_matrix

    logger.info(f"Fusing {len(scans)} scans with voxel_size={voxel_size}")

    fused_cloud = []

    for i, scan in enumerate(scans):
        if i >= len(optimized_poses):
            logger.warning(
                f"More scans ({len(scans)}) than poses ({len(optimized_poses)}), stopping"
            )
            break

        pose = optimized_poses[i]
        pos = pose[1:4]
        quat = pose[4:8]

        # Transform scan to world frame
        R = quaternion_to_rotation_matrix(quat)
        scan_world = scan @ R.T + pos[np.newaxis, :]

        fused_cloud.append(scan_world)

    # Concatenate all clouds
    if not fused_cloud:
        logger.warning("No scans to fuse")
        return np.array([]).reshape(0, 3)

    fused_cloud = np.vstack(fused_cloud)
    logger.info(f"Fused cloud: {len(fused_cloud)} points")

    # Downsample using voxel grid
    fused_cloud = voxel_downsample(fused_cloud, voxel_size)
    logger.info(f"After downsampling: {len(fused_cloud)} points")

    return fused_cloud


def voxel_downsample(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Downsample point cloud using voxel grid.

    Args:
        cloud: Nx3 point cloud
        voxel_size: Voxel size (meters)

    Returns:
        Downsampled Mx3 point cloud
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd_ds = pcd.voxel_down_sample(voxel_size)

        return np.asarray(pcd_ds.points, dtype=np.float32)

    except Exception as e:
        logger.warning(f"Open3D downsampling failed: {e}, using fallback")
        return voxel_downsample_naive(cloud, voxel_size)


def voxel_downsample_naive(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Naive voxel downsampling (fallback if Open3D unavailable).
    """
    # Round to voxel grid
    grid_points = np.round(cloud / voxel_size).astype(np.int32)

    # Get unique voxels
    unique_voxels, indices = np.unique(grid_points, axis=0, return_index=True)

    # Return one point per voxel (the first point in that voxel)
    return cloud[indices]


def colorize_cloud(
    cloud: np.ndarray,
    scans: List[np.ndarray],
    optimized_poses: np.ndarray,
    images: List[np.ndarray],
    intrinsics: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Color a fused point cloud by projecting camera images.

    Args:
        cloud: Mx3 fused point cloud in world frame
        scans: Original scans (for reprojection)
        optimized_poses: Nx7 optimized poses
        images: List of camera images
        intrinsics: Dict of camera intrinsics {camera_name: {fx, fy, cx, cy, ...}}

    Returns:
        Tuple of (colored_cloud Mx3, colors Mx3 RGB)

    TODO: implement full camera projection with visibility testing
    """
    logger.warning("Point cloud colorization not yet fully implemented")

    # Placeholder: use height-based coloring
    colors = np.zeros_like(cloud, dtype=np.uint8)
    z_min = cloud[:, 2].min()
    z_max = cloud[:, 2].max()

    if z_max > z_min:
        z_normalized = (cloud[:, 2] - z_min) / (z_max - z_min)
        colors[:, 0] = (z_normalized * 255).astype(np.uint8)  # R
        colors[:, 1] = ((1 - z_normalized) * 255).astype(np.uint8)  # G
        colors[:, 2] = 128  # B

    return cloud, colors
