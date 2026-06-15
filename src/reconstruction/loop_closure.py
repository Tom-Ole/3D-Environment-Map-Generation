"""
Loop closure detection via spatial proximity + ICP verification.

Algorithm (matches KISS-SLAM / LOAM-style systems):
  1. Build a KD-tree over keyframe positions.
  2. For each keyframe i, query neighbours within `threshold` metres.
  3. Keep only candidates j that are at least MIN_FRAME_SEPARATION keyframes
     earlier (prevents short-range odometry edges being treated as loops).
  4. Verify each candidate with point-to-point ICP initialised from the
     odometry relative pose.  Accept if ICP fitness ≥ MIN_ICP_FITNESS.
"""

import logging
from typing import Callable, List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from reconstruction.types import LoopEdge

logger = logging.getLogger(__name__)

_MIN_FRAME_SEPARATION = 20   # keyframe gap to consider a loop (not just adjacent odom)
_MIN_ICP_FITNESS = 0.25      # minimum ICP overlap ratio [0–1]


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
        keyframe_poses: List of 4x4 odometry poses (world frame).
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

    # Cache of loaded clouds (lazy, keyed by keyframe index)
    cloud_cache: dict[int, o3d.geometry.PointCloud] = {}

    def _get_cloud(idx: int) -> o3d.geometry.PointCloud:
        if idx not in cloud_cache:
            cloud_cache[idx] = _load_and_downsample(keyframe_paths[idx], voxel_size)
        return cloud_cache[idx]

    loop_edges: List[LoopEdge] = []

    for i in range(n):
        candidates = tree.query_ball_point(positions[i], r=threshold)
        candidates = [j for j in candidates if j < i - _MIN_FRAME_SEPARATION]

        for j in candidates:
            T_init = np.linalg.inv(keyframe_poses[j]) @ keyframe_poses[i]

            source = _get_cloud(i)
            target = _get_cloud(j)

            if len(source.points) < 10 or len(target.points) < 10:
                continue

            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                max_correspondence_distance,
                T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=icp_iterations
                ),
            )

            if result.fitness >= _MIN_ICP_FITNESS:
                info = np.eye(6) * (result.fitness * 100.0)
                loop_edges.append(
                    LoopEdge(
                        source_idx=i,
                        target_idx=j,
                        T_source_to_target=result.transformation.copy(),
                        fitness=result.fitness,
                        information=info,
                    )
                )
                logger.debug(f"Loop: kf{i} → kf{j}  fitness={result.fitness:.3f}")

        if progress_cb:
            progress_cb(i + 1, n)

    logger.info(f"Loop closure: {len(loop_edges)} verified edges from {n} keyframes")
    return loop_edges


def _load_and_downsample(path, voxel_size: float) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        return pcd
    pcd = pcd.voxel_down_sample(max(voxel_size, 0.05))
    return pcd
