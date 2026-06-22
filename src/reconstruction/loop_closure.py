"""
Loop closure detection via spatial proximity + ICP verification.

Algorithm:
  1. Build a KD-tree over keyframe positions.
  2. For each keyframe i, query neighbours within `threshold` metres.
  3. Discard candidates j that are fewer than MIN_FRAME_SEPARATION keyframes
     earlier (prevents odometry adjacency being treated as loops).
  4. Verify each candidate with point-to-plane ICP initialised from the
     odometry relative pose.  Accept if ICP fitness ≥ MIN_ICP_FITNESS.
  5. Cap loop edges per keyframe at MAX_LOOPS_PER_FRAME to avoid over-connection.

Improvements over the original:
  - Point-to-plane ICP (more accurate than point-to-point for smooth surfaces).
  - Stricter fitness threshold (0.35 vs 0.25) to reduce false positives.
  - Information matrix calibrated by fitness × sqrt(inlier_count) so stronger
    verifications contribute more to pose graph optimisation.
  - Per-keyframe loop cap prevents hub keyframes from dominating the graph.
"""

import logging
from typing import Callable, List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from reconstruction.types import LoopEdge

logger = logging.getLogger(__name__)

_MIN_FRAME_SEPARATION = 20   # keyframe gap before a candidate is considered a loop
_MIN_ICP_FITNESS = 0.35      # ICP overlap ratio threshold (higher = fewer false loops)
_MAX_LOOPS_PER_FRAME = 3     # max loop edges per keyframe (prevents over-connection)
_NORMAL_RADIUS_MULT = 3.0    # normal search radius = voxel_size × this factor


def detect_loop_closures(
    keyframe_poses: List[np.ndarray],
    keyframe_paths,
    voxel_size: float = 0.1,
    threshold: float = 2.0,
    max_correspondence_distance: float = 0.3,
    icp_iterations: int = 50,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[LoopEdge]:
    """
    Detect and ICP-verify loop closure candidates.

    Args:
        keyframe_poses: List of 4×4 odometry poses (world frame).
        keyframe_paths: List of Path objects to PLY scans (same order).
        voxel_size: Downsample resolution for ICP clouds (m).
        threshold: Spatial search radius for candidate pairs (m).
        max_correspondence_distance: ICP max correspondence distance (m).
        icp_iterations: ICP iteration cap for verification.
        progress_cb: Called with (done, total).

    Returns:
        List of verified LoopEdge objects.
    """
    n = len(keyframe_poses)
    if n < _MIN_FRAME_SEPARATION + 1:
        logger.info("Too few keyframes for loop closure detection")
        return []

    positions = np.array([T[:3, 3] for T in keyframe_poses])
    tree = cKDTree(positions)

    normal_radius = voxel_size * _NORMAL_RADIUS_MULT

    # Lazy cloud cache keyed by keyframe index
    cloud_cache: dict[int, o3d.geometry.PointCloud] = {}

    def _get_cloud(idx: int) -> o3d.geometry.PointCloud:
        if idx not in cloud_cache:
            cloud_cache[idx] = _load_downsample_normals(
                keyframe_paths[idx], voxel_size, normal_radius
            )
        return cloud_cache[idx]

    loop_edges: List[LoopEdge] = []
    loops_per_frame: dict[int, int] = {}

    for i in range(n):
        candidates = tree.query_ball_point(positions[i], r=threshold)
        candidates = [j for j in candidates if j < i - _MIN_FRAME_SEPARATION]
        # Prioritise candidates closest in position to maximise true-positive rate
        candidates.sort(key=lambda j: np.linalg.norm(positions[i] - positions[j]))

        for j in candidates:
            if loops_per_frame.get(i, 0) >= _MAX_LOOPS_PER_FRAME:
                break

            T_init = np.linalg.inv(keyframe_poses[j]) @ keyframe_poses[i]
            source = _get_cloud(i)
            target = _get_cloud(j)

            if len(source.points) < 10 or len(target.points) < 10:
                continue

            result = _run_icp(source, target, T_init, max_correspondence_distance, icp_iterations)

            if result.fitness >= _MIN_ICP_FITNESS:
                info = _build_info_matrix(result, source)
                loop_edges.append(
                    LoopEdge(
                        source_idx=i,
                        target_idx=j,
                        T_source_to_target=result.transformation.copy(),
                        fitness=result.fitness,
                        information=info,
                    )
                )
                loops_per_frame[i] = loops_per_frame.get(i, 0) + 1
                logger.debug(
                    f"Loop: kf{i} → kf{j}  fitness={result.fitness:.3f}  "
                    f"rmse={result.inlier_rmse:.4f}"
                )

        if progress_cb:
            progress_cb(i + 1, n)

    logger.info(f"Loop closure: {len(loop_edges)} verified edges from {n} keyframes")
    return loop_edges


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    max_corr: float,
    max_iter: int,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Run point-to-plane ICP if both clouds have normals, otherwise point-to-point.

    Point-to-plane ICP converges faster and is more accurate on smooth surfaces,
    which is typical for LiDAR scans of structured indoor environments.
    """
    use_p2plane = (
        source.has_normals()
        and target.has_normals()
        and len(source.normals) == len(source.points)
        and len(target.normals) == len(target.points)
    )

    if use_p2plane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_corr,
        T_init,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )


def _build_info_matrix(
    result: o3d.pipelines.registration.RegistrationResult,
    source: o3d.geometry.PointCloud,
) -> np.ndarray:
    """
    Build a 6×6 information matrix for a loop closure edge.

    The weight is proportional to fitness × sqrt(inlier_count) so that:
    - Higher overlap (fitness) → more confident
    - More inlier correspondences → more confident
    The square-root dampens the effect of very large clouds.
    """
    n_inliers = max(1, int(result.fitness * len(source.points)))
    weight = result.fitness * np.sqrt(n_inliers) * 5.0
    return np.eye(6) * weight


def _load_downsample_normals(
    path,
    voxel_size: float,
    normal_radius: float,
) -> o3d.geometry.PointCloud:
    """Load, voxel-downsample, and estimate normals for ICP."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        return pcd
    pcd = pcd.voxel_down_sample(max(voxel_size, 0.05))
    if len(pcd.points) < 10:
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )
    return pcd
