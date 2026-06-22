"""
KISS-ICP odometry frontend.

Wraps the kiss_icp Python package (Vizzo et al., RA-L 2023).
Each raw point cloud (Nx3 float32, sensor frame) is fed sequentially
into KissICP.register_frame(); the resulting pose list is returned.

KISS-ICP handles:
  - Adaptive voxel downsampling
  - Point-to-point ICP with robust Welsch loss
  - Adaptive correspondence-distance threshold

When `vio_poses` is supplied (one 4×4 SE(3) per scan, in any consistent
world frame), the relative VIO motion is used as a fallback pose update
whenever a scan is degenerate (< 10 points) instead of copying the last
SLAM pose verbatim.

Reference: https://github.com/PRBonn/kiss-icp
"""

import logging
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def run_odometry(
    scan_frames,
    load_cloud_fn: Callable,
    voxel_size: float = 1.0,
    max_range: float = 100.0,
    min_range: float = 0.5,
    vio_poses: Optional[List[np.ndarray]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[np.ndarray]:
    """
    Run KISS-ICP on a sequence of LiDAR scans.

    Args:
        scan_frames: List of ScanFrame objects.
        load_cloud_fn: Callable(path) → Nx3 float32 (sensor frame, range-filtered).
        voxel_size: KISS-ICP internal voxel-map resolution (metres).
                    Governs downsampling inside KISS-ICP; typically 0.5–2.0 m.
        max_range / min_range: Range gates forwarded to KISSConfig.
        vio_poses: Optional list of 4×4 SE(3) matrices (one per scan frame) in
                   LiDAR-world frame derived from SPOT VIO + body→lidar extrinsic.
                   Used only as a fallback when a scan is degenerate; does not
                   override KISS-ICP's own pose estimates for normal frames.
        progress_cb: Optional callable(done, total).

    Returns:
        List of 4×4 SE(3) numpy arrays – one per scan frame, in world frame.
    """
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    cfg = KISSConfig()
    cfg.data.max_range = max_range
    cfg.data.min_range = min_range
    cfg.mapping.voxel_size = voxel_size

    odometry = KissICP(config=cfg)
    poses: List[np.ndarray] = []
    n = len(scan_frames)
    n_vio_fallbacks = 0

    for i, frame in enumerate(scan_frames):
        pts = load_cloud_fn(frame.path)

        if pts is None or len(pts) < 10:
            poses.append(_degenerate_pose(i, poses, vio_poses))
            n_vio_fallbacks += 1
        else:
            odometry.register_frame(pts, np.array([]))
            poses.append(odometry.last_pose.copy())

        if progress_cb:
            progress_cb(i + 1, n)

    if n_vio_fallbacks:
        logger.info(
            f"KISS-ICP: {len(poses)} poses from {n} scans "
            f"({n_vio_fallbacks} degenerate frames used VIO fallback)"
        )
    else:
        logger.info(f"KISS-ICP: {len(poses)} poses from {n} scans")
    return poses


def _degenerate_pose(
    i: int,
    poses: List[np.ndarray],
    vio_poses: Optional[List[np.ndarray]],
) -> np.ndarray:
    """
    Return best available pose estimate when scan i is degenerate.

    Strategy (in priority order):
    1. If VIO relative motion is available for frame i-1 → i, apply it to
       the last SLAM pose.  This propagates VIO motion without requiring the
       two coordinate frames to be aligned absolutely.
    2. Copy the last SLAM pose (constant-position fallback).
    3. Identity if no poses exist yet.
    """
    if not poses:
        return np.eye(4)

    if (
        vio_poses is not None
        and i > 0
        and i < len(vio_poses)
        and vio_poses[i] is not None
        and vio_poses[i - 1] is not None
    ):
        T_rel = np.linalg.inv(vio_poses[i - 1]) @ vio_poses[i]
        return poses[-1] @ T_rel

    return poses[-1].copy()
