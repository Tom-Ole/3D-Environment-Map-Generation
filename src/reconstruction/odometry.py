"""KISS-ICP based LiDAR odometry with SPOT pose seeding."""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# If KISS-ICP's estimated position drifts more than this distance from the
# corresponding SPOT pose, reset KISS-ICP's last known pose from SPOT.
# This catches divergence on sharp turns or in featureless corridors.
_DIVERGENCE_RESET_THRESHOLD_M = 1.5


def kiss_icp_odometry(
    scans: List[np.ndarray],
    scan_poses: Optional[np.ndarray] = None,
    max_distance: float = 0.1,
    icp_iterations: int = 10,
    voxel_size: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run KISS-ICP odometry on a sequence of point clouds.

    Args:
        scans: List of Nx3 point clouds (in LiDAR sensor frame)
        scan_poses: Nx7 per-scan SPOT poses in LiDAR frame [x,y,z,qx,qy,qz,qw].
                    Already has the body→lidar extrinsic applied.
                    Used to warm-start KISS-ICP and catch divergence.
        max_distance: Max ICP correspondence distance (meters)
        icp_iterations: ICP iterations per frame (unused directly; KISS manages its own)
        voxel_size: Voxel size for KISS-ICP's internal map

    Returns:
        Tuple of:
        - Odometry poses Nx7 [x, y, z, qx, qy, qz, qw]
        - Frame indices N array
    """
    if not scans:
        return np.array([]), np.array([])

    try:
        from kiss_icp.pipeline import KissICP
        from kiss_icp.config import KISSConfig
        from scipy.spatial.transform import Rotation
        from reconstruction.global_opt import transform_7d_to_4x4, transform_4x4_to_7d

        logger.info(f"Running KISS-ICP on {len(scans)} scans")

        config = KISSConfig()
        config.mapping.voxel_size = voxel_size
        # deskew requires per-point timestamps which the Velodyne SPOT API does
        # not expose, so we leave it disabled.
        config.data.deskew = False
        odometry = KissICP(config=config)

        # ── Warm-start ────────────────────────────────────────────────────────
        # Pre-populate KISS-ICP's pose list with the SPOT pose for frame 0.
        # This places the map in world coordinates from the very first frame
        # rather than starting at the sensor-frame origin (identity).
        if scan_poses is not None and len(scan_poses) > 0:
            T0 = transform_7d_to_4x4(scan_poses[0])
            odometry.poses.append(T0)
            logger.info("Warm-started KISS-ICP from first SPOT pose")

        poses = []
        frame_indices = []
        reset_count = 0

        for idx, scan in enumerate(scans):
            # Remove NaN / Inf and zero-range returns.
            valid = np.isfinite(scan).all(axis=1)
            scan_clean = scan[valid]
            dist = np.linalg.norm(scan_clean, axis=1)
            scan_clean = scan_clean[dist > 0.1]

            if len(scan_clean) == 0:
                logger.warning(f"Scan {idx} has no valid points, skipping")
                prev = poses[-1] if poses else np.zeros(7)
                if len(prev) == 0:
                    prev = np.array([0., 0., 0., 0., 0., 0., 1.])
                poses.append(prev)
                frame_indices.append(idx)
                continue

            # ── Per-frame divergence check ─────────────────────────────────
            # If we have a SPOT reference for this frame and KISS-ICP's last
            # pose has strayed too far, reset KISS-ICP to the SPOT pose.
            # This corrects drift without replacing the local ICP registration.
            if (
                scan_poses is not None
                and idx < len(scan_poses)
                and odometry.poses
            ):
                T_spot = transform_7d_to_4x4(scan_poses[idx])
                T_kiss = odometry.poses[-1]
                drift = np.linalg.norm(T_kiss[:3, 3] - T_spot[:3, 3])
                if drift > _DIVERGENCE_RESET_THRESHOLD_M:
                    logger.warning(
                        f"Frame {idx}: KISS-ICP drifted {drift:.2f} m from SPOT, resetting"
                    )
                    odometry.poses[-1] = T_spot
                    reset_count += 1

            timestamps = np.zeros(len(scan_clean))
            odometry.register_frame(scan_clean, timestamps)

            T = odometry.last_pose
            pose_7d = transform_4x4_to_7d(T)
            poses.append(pose_7d)
            frame_indices.append(idx)

            if (idx + 1) % 50 == 0:
                logger.debug(f"Processed {idx + 1}/{len(scans)} scans")

        if reset_count:
            logger.info(f"KISS-ICP divergence resets: {reset_count}")

        logger.info(f"KISS-ICP complete: {len(poses)} poses")
        return np.array(poses), np.array(frame_indices)

    except ImportError:
        logger.warning("kiss_icp not installed, falling back to SPOT poses")
        return _spot_pose_fallback(scans, scan_poses)


def apply_body_to_lidar_extrinsic(
    body_poses: np.ndarray,
    body_to_lidar: np.ndarray,
) -> np.ndarray:
    """
    Convert SPOT body-frame poses to LiDAR-frame poses by composing with the
    body→lidar extrinsic transform.

    All SPOT poses (from the vision frame) describe where the robot *body* is
    in the world.  The Velodyne sensor is offset from the body; applying this
    extrinsic gives T_world←lidar for each scan, which is what KISS-ICP and
    the fusion step need.

    Args:
        body_poses: Nx7 SPOT body poses [x, y, z, qx, qy, qz, qw]
        body_to_lidar: 4×4 transform: T_body←lidar  (lidar frame → body frame)

    Returns:
        Nx7 LiDAR-frame poses [x, y, z, qx, qy, qz, qw]
    """
    from reconstruction.global_opt import transform_7d_to_4x4, transform_4x4_to_7d

    lidar_poses = []
    for pose_7d in body_poses:
        T_world_from_body = transform_7d_to_4x4(pose_7d)
        T_world_from_lidar = T_world_from_body @ body_to_lidar
        lidar_poses.append(transform_4x4_to_7d(T_world_from_lidar))

    return np.array(lidar_poses)


def _spot_pose_fallback(
    scans: List[np.ndarray],
    scan_poses: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Use SPOT poses directly when KISS-ICP is unavailable."""
    logger.warning("Using SPOT poses as odometry (kiss_icp not installed)")

    if scan_poses is not None and len(scan_poses) == len(scans):
        return scan_poses, np.arange(len(scan_poses))

    # Last resort: stationary (all identity).
    poses = np.tile(np.array([0., 0., 0., 0., 0., 0., 1.]), (len(scans), 1))
    return poses, np.arange(len(scans))


def downsample_cloud(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-downsample a point cloud (Nx3)."""
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        return np.asarray(pcd.voxel_down_sample(voxel_size).points)

    except Exception as e:
        logger.warning(f"Open3D downsampling failed: {e}")
        n = max(1, int(len(cloud) * (voxel_size / 0.05) ** 3))
        idx = np.random.choice(len(cloud), size=min(n, len(cloud)), replace=False)
        return cloud[idx]
