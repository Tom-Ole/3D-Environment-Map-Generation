"""Loop closure detection and registration."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from reconstruction.types import LoopClosureCandidate, LoopClosureResult

logger = logging.getLogger(__name__)


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
        distance_threshold: Max distance for candidates (meters)
        min_frame_gap: Min frames between candidate pairs

    Returns:
        List of LoopClosureCandidate objects
    """
    from scipy.spatial import cKDTree

    candidates = []
    positions = poses[:, 1:4]

    logger.info(
        f"Detecting loop closures (threshold: {distance_threshold}m, gap: {min_frame_gap})"
    )

    tree = cKDTree(positions)
    pairs = tree.query_pairs(distance_threshold)

    # Group by source frame, keep closest max_candidates_per_frame per frame
    from collections import defaultdict
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


def register_scan_pair(
    scan1: np.ndarray,
    scan2: np.ndarray,
    initial_transform: Optional[np.ndarray] = None,
    max_correspondence_distance: float = 0.1,
    iterations: int = 50,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Register two point clouds using ICP (via small_gicp or Open3D).

    Args:
        scan1: Source Nx3 point cloud
        scan2: Target Nx3 point cloud
        initial_transform: 4x4 initial transformation guess (optional)
        max_correspondence_distance: Max correspondence distance (meters)
        iterations: Number of ICP iterations

    Returns:
        Tuple of (success, 7D transform [x, y, z, qx, qy, qz, qw])

    TODO: integrate small_gicp for faster registration
    """
    try:
        import open3d as o3d

        # Create point clouds
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(scan1)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(scan2)

        # Initial transform (identity if not provided)
        if initial_transform is None:
            initial_transform = np.eye(4)

        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            pcd1,
            pcd2,
            max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iterations
            ),
        )

        if result.fitness < 0.1:
            logger.warning(f"Low fitness in registration: {result.fitness}")
            return False, None

        # Extract transform as 7D vector
        T = result.transformation
        pos = T[:3, 3]
        quat = matrix_to_quaternion(T[:3, :3])

        transform_7d = np.array(
            [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
        )

        logger.debug(f"ICP registration: fitness={result.fitness:.4f}")
        return True, transform_7d

    except Exception as e:
        logger.error(f"Failed to register scan pair: {e}")
        return False, None


def process_loop_closures(
    candidates: List[LoopClosureCandidate],
    scans: List[np.ndarray],
    poses: np.ndarray,
    max_correspondence_distance: float = 0.1,
) -> LoopClosureResult:
    """
    Process loop closure candidates and register valid pairs.

    Args:
        candidates: List of LoopClosureCandidate objects
        scans: List of point clouds
        poses: Nx7 odometry poses
        max_correspondence_distance: Max ICP correspondence distance

    Returns:
        LoopClosureResult with registered pairs
    """
    result = LoopClosureResult(candidates=candidates, registered_pairs={})

    for candidate in candidates:
        src_idx = candidate.source_idx
        tgt_idx = candidate.target_idx

        if src_idx >= len(scans) or tgt_idx >= len(scans):
            continue

        # Estimate initial transform from pose difference
        src_pos = poses[src_idx, 1:4]
        tgt_pos = poses[tgt_idx, 1:4]
        initial_pos = tgt_pos - src_pos

        # Try registration
        success, transform_7d = register_scan_pair(
            scans[src_idx],
            scans[tgt_idx],
            max_correspondence_distance=max_correspondence_distance,
        )

        if success and transform_7d is not None:
            result.registered_pairs[(src_idx, tgt_idx)] = transform_7d
            result.loop_count += 1
            logger.debug(
                f"Loop closure registered: {src_idx} -> {tgt_idx}"
            )

    logger.info(f"Registered {result.loop_count} loop closures")
    return result


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (scalar-last: [x, y, z, w])."""
    trace = np.trace(R)

    if trace > 0:
        S = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([x, y, z, w])
