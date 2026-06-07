"""Point cloud and mesh colorization."""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def colorize_by_height(
    cloud: np.ndarray,
) -> np.ndarray:
    """
    Simple height-based coloring (red low, blue high).

    Args:
        cloud: Mx3 point cloud

    Returns:
        Mx3 RGB colors
    """
    colors = np.zeros_like(cloud, dtype=np.uint8)
    z_min = cloud[:, 2].min()
    z_max = cloud[:, 2].max()

    if z_max > z_min:
        z_normalized = (cloud[:, 2] - z_min) / (z_max - z_min)
        colors[:, 0] = (z_normalized * 255).astype(np.uint8)  # R
        colors[:, 1] = 128
        colors[:, 2] = ((1 - z_normalized) * 255).astype(np.uint8)  # B

    return colors


def colorize_by_camera_projection(
    cloud: np.ndarray,
    scans: List[np.ndarray],
    optimized_poses: np.ndarray,
    images: List[np.ndarray],
    intrinsics: dict,
    camera_poses: Optional[dict] = None,
) -> np.ndarray:
    """
    Colorize point cloud by projecting camera images.

    TODO: Full implementation with visibility testing and occlusion handling.

    Args:
        cloud: Mx3 point cloud in world frame
        scans: Original scans
        optimized_poses: Optimized scan poses
        images: Camera images
        intrinsics: Camera intrinsics
        camera_poses: Optional camera extrinsics (sensor-to-body transforms)

    Returns:
        Mx3 RGB colors (or height-based fallback if unavailable)
    """
    logger.warning("Camera-based colorization not yet implemented, using height-based")

    return colorize_by_height(cloud)


def colorize_mesh(
    mesh,
    cloud: np.ndarray,
    colors: np.ndarray,
) -> None:
    """
    Color a mesh by projecting vertex colors from a colored point cloud.

    Args:
        mesh: Open3D TriangleMesh object (modified in-place)
        cloud: Mx3 colored reference cloud
        colors: Mx3 RGB colors
    """
    try:
        import open3d as o3d

        mesh.vertex_colors = o3d.utility.Vector3dVector(
            colors.astype(np.float32) / 255.0
        )
        logger.info("Colored mesh vertices")

    except Exception as e:
        logger.error(f"Failed to color mesh: {e}")
