"""
KISS-ICP odometry frontend.

Wraps the kiss_icp Python package (Vizzo et al., RA-L 2023).
Each raw point cloud (Nx3 float32, sensor frame) is fed sequentially
into KissICP.register_frame(); the resulting pose list is returned.

KISS-ICP handles:
  - Adaptive voxel downsampling
  - Point-to-point ICP with robust Welsch loss
  - Adaptive correspondence-distance threshold

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
        progress_cb: Optional callable(done, total).

    Returns:
        List of 4x4 SE(3) numpy arrays – one per scan frame, in world frame.
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

    for i, frame in enumerate(scan_frames):
        pts = load_cloud_fn(frame.path)

        if pts is None or len(pts) < 10:
            # Degenerate scan – propagate last known pose
            poses.append(poses[-1].copy() if poses else np.eye(4))
        else:
            odometry.register_frame(pts, np.array([]))
            poses.append(odometry.last_pose.copy())

        if progress_cb:
            progress_cb(i + 1, n)

    logger.info(f"KISS-ICP: {len(poses)} poses from {n} scans")
    return poses
