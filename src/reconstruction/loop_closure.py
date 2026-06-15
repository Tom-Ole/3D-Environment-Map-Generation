"""Loop closure detection and registration."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from reconstruction.types import LoopClosureCandidate, LoopClosureResult

logger = logging.getLogger(__name__)

# Minimum ICP fitness to accept a loop closure registration.
_MIN_FITNESS = 0.3


def detect_loop_closures(
    poses: np.ndarray,
    scans: List[np.ndarray],
    distance_threshold: float = 2.0,
    min_frame_gap: int = 10,
    max_candidates_per_frame: int = 3,
) -> List[LoopClosureCandidate]:
    """
    Detect loop closure candidates using spatial proximity of poses.

    Args:
        poses: Nx7 odometry poses [t, x, y, z, qx, qy, qz, qw]
        scans: List of Nx3 point clouds
        distance_threshold: Max distance between candidate poses (meters)
        min_frame_gap: Minimum frame separation between candidate pairs
        max_candidates_per_frame: Keep at most this many candidates per source frame

    Returns:
        List of LoopClosureCandidate objects
    """
    from scipy.spatial import cKDTree
    from collections import defaultdict

    candidates = []
    positions = poses[:, 1:4]

    logger.info(
        f"Detecting loop closures (threshold={distance_threshold}m, gap={min_frame_gap})"
    )

    tree = cKDTree(positions)
    pairs = tree.query_pairs(distance_threshold)

    per_frame: dict = defaultdict(list)
    for i, j in pairs:
        if abs(j - i) < min_frame_gap:
            continue
        distance = np.linalg.norm(positions[i] - positions[j])
        per_frame[i].append((distance, j))

    for i, neighbours in per_frame.items():
        neighbours.sort()
        for distance, j in neighbours[:max_candidates_per_frame]:
            candidates.append(LoopClosureCandidate(
                source_idx=i,
                target_idx=j,
                distance=distance,
                confidence=1.0 - (distance / distance_threshold),
            ))

    logger.info(f"Found {len(candidates)} loop closure candidates")
    return candidates


def _make_o3d_pcd(pts: np.ndarray, voxel_size: float):
    """Downsample pts and estimate normals; returns an Open3D PointCloud."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd


def _fpfh_feature(pcd, voxel_size: float):
    """Compute FPFH features for global registration."""
    import open3d as o3d

    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )


def _ransac_global_registration(src_pcd, tgt_pcd, src_fpfh, tgt_fpfh, voxel_size: float):
    """RANSAC-based global registration using FPFH features."""
    import open3d as o3d

    dist = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd,
        tgt_pcd,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
    )
    return result


def register_scan_pair(
    scan1: np.ndarray,
    scan2: np.ndarray,
    initial_transform: Optional[np.ndarray] = None,
    max_correspondence_distance: float = 0.1,
    iterations: int = 50,
    voxel_size: float = 0.05,
    use_fpfh_fallback: bool = True,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Register two point clouds with Point-to-Plane ICP.

    Uses FPFH + RANSAC as a fallback initial alignment when the pose-seeded
    guess appears unreliable (fitness below threshold after a first ICP pass).

    Args:
        scan1: Source Nx3 point cloud
        scan2: Target Nx3 point cloud
        initial_transform: 4×4 initial transformation guess (optional)
        max_correspondence_distance: ICP correspondence distance (meters)
        iterations: Number of ICP iterations
        voxel_size: Voxel size used for downsampling and FPFH radius
        use_fpfh_fallback: Try FPFH + RANSAC if pose-seeded ICP has low fitness

    Returns:
        Tuple of (success, transform_7d [x,y,z,qx,qy,qz,qw], information_6x6)
    """
    import open3d as o3d

    src_pcd = _make_o3d_pcd(scan1, voxel_size)
    tgt_pcd = _make_o3d_pcd(scan2, voxel_size)

    if initial_transform is None:
        initial_transform = np.eye(4)

    icp_p2plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, max_correspondence_distance,
        initial_transform, icp_p2plane, criteria,
    )

    # If the pose-seeded ICP gives poor fitness, try FPFH global registration
    # as a fresh initial alignment and re-run ICP from there.
    if result.fitness < _MIN_FITNESS and use_fpfh_fallback:
        logger.debug("Pose-seeded ICP low fitness, trying FPFH global registration")
        try:
            src_fpfh = _fpfh_feature(src_pcd, voxel_size)
            tgt_fpfh = _fpfh_feature(tgt_pcd, voxel_size)
            ransac_result = _ransac_global_registration(
                src_pcd, tgt_pcd, src_fpfh, tgt_fpfh, voxel_size
            )
            if ransac_result.fitness > 0.0:
                result = o3d.pipelines.registration.registration_icp(
                    src_pcd, tgt_pcd, max_correspondence_distance,
                    ransac_result.transformation, icp_p2plane, criteria,
                )
        except Exception as e:
            logger.debug(f"FPFH fallback failed: {e}")

    if result.fitness < _MIN_FITNESS:
        logger.debug(f"Registration rejected: fitness={result.fitness:.3f}")
        return False, None, None

    # Compute a proper 6×6 information matrix from point cloud overlap.
    # This gives the optimizer a data-driven confidence for this edge rather
    # than the previous fixed scalar multiple of identity.
    try:
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_pcd, tgt_pcd, max_correspondence_distance, result.transformation
        )
    except Exception:
        info = np.eye(6) * result.fitness

    T = result.transformation
    from scipy.spatial.transform import Rotation
    pos = T[:3, 3]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
    transform_7d = np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])

    logger.debug(f"ICP registered: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f}")
    return True, transform_7d, info


def process_loop_closures(
    candidates: List[LoopClosureCandidate],
    scans: List[np.ndarray],
    poses: np.ndarray,
    max_correspondence_distance: float = 0.1,
    voxel_size: float = 0.05,
) -> LoopClosureResult:
    """
    Register all loop closure candidates and collect transforms + information matrices.

    Args:
        candidates: List of LoopClosureCandidate objects
        scans: List of point clouds
        poses: Nx7 or Nx8 odometry poses
        max_correspondence_distance: Max ICP correspondence distance
        voxel_size: Voxel size for downsampling / FPFH radius

    Returns:
        LoopClosureResult with registered_pairs and information_matrices populated
    """
    from reconstruction.global_opt import transform_7d_to_4x4, relative_pose as _rel

    # Strip optional timestamp column so we always work with 7D [x,y,z,q…]
    if poses.shape[1] == 8:
        pose_data = poses[:, 1:]
    else:
        pose_data = poses

    result = LoopClosureResult(candidates=candidates, registered_pairs={}, information_matrices={})

    for candidate in candidates:
        src_idx = candidate.source_idx
        tgt_idx = candidate.target_idx

        if src_idx >= len(scans) or tgt_idx >= len(scans):
            continue
        if src_idx >= len(pose_data) or tgt_idx >= len(pose_data):
            continue

        # Build initial guess: transform that maps source scan into target frame.
        # relative_pose(A, B) = inv(A) * B, so relative_pose(tgt, src) gives
        # T_{tgt←src} which maps source-frame points into target frame — exactly
        # what ICP needs as its initial transform.
        rel_7d = _rel(pose_data[tgt_idx], pose_data[src_idx])
        initial_transform = transform_7d_to_4x4(rel_7d)

        success, transform_7d, info = register_scan_pair(
            scans[src_idx],
            scans[tgt_idx],
            initial_transform=initial_transform,
            max_correspondence_distance=max_correspondence_distance,
            voxel_size=voxel_size,
        )

        if success and transform_7d is not None:
            result.registered_pairs[(src_idx, tgt_idx)] = transform_7d
            result.information_matrices[(src_idx, tgt_idx)] = info
            result.loop_count += 1
            logger.debug(f"Loop closure registered: {src_idx} → {tgt_idx}")

    logger.info(f"Registered {result.loop_count}/{len(candidates)} loop closures")
    return result


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion (scalar-last: [x, y, z, w])."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()
