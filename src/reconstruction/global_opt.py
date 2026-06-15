"""
Pose-graph optimisation using Open3D's Levenberg-Marquardt solver.

Graph structure:
  Nodes  — one per keyframe, initialised from KISS-ICP odometry pose.
  Edges  — (a) sequential odometry edges (high confidence, uncertain=False)
           (b) loop closure edges   (lower confidence, uncertain=True)

Reference: Choi et al., "Robust Reconstruction of Indoor Scenes",
           CVPR 2015.  Implemented via open3d.pipelines.registration.
"""

import logging
from typing import List

import numpy as np
import open3d as o3d

from reconstruction.types import LoopEdge

logger = logging.getLogger(__name__)

_ODOM_INFO_WEIGHT = 500.0       # diagonal weight for consecutive odometry edges
_OPT_MAX_ITER = 1000            # LM max iterations
_OPT_EDGE_PRUNE = 0.25          # prune loop edges whose residual exceeds this
_OPT_MAX_CORR_DIST = 0.3        # used only by the pruning heuristic


def optimize_pose_graph(
    keyframe_poses: List[np.ndarray],
    loop_edges: List[LoopEdge],
) -> List[np.ndarray]:
    """
    Build and optimize a pose graph from odometry + loop closure edges.

    Args:
        keyframe_poses: Initial 4x4 poses from KISS-ICP (world frame).
        loop_edges: Verified LoopEdge objects from loop_closure module.

    Returns:
        Refined list of 4x4 SE(3) poses (same length as input).
    """
    if len(keyframe_poses) < 2:
        return list(keyframe_poses)

    pose_graph = o3d.pipelines.registration.PoseGraph()

    # --- Nodes ---
    for T in keyframe_poses:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T.copy()))

    # --- Odometry edges (sequential, high-confidence) ---
    odom_info = np.eye(6) * _ODOM_INFO_WEIGHT
    for i in range(len(keyframe_poses) - 1):
        T_rel = np.linalg.inv(keyframe_poses[i]) @ keyframe_poses[i + 1]
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=i,
                target_node_id=i + 1,
                transformation=T_rel,
                information=odom_info,
                uncertain=False,
            )
        )

    # --- Loop closure edges (uncertain) ---
    for edge in loop_edges:
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=edge.source_idx,
                target_node_id=edge.target_idx,
                transformation=edge.T_source_to_target,
                information=edge.information,
                uncertain=True,
            )
        )

    logger.info(
        f"Pose graph: {len(pose_graph.nodes)} nodes, "
        f"{len(pose_graph.edges)} edges "
        f"({len(loop_edges)} loop closures)"
    )

    # --- Optimize ---
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=_OPT_MAX_CORR_DIST,
        edge_prune_threshold=_OPT_EDGE_PRUNE,
        reference_node=0,
    )
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = _OPT_MAX_ITER

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        criteria,
        option,
    )

    optimized = [np.asarray(node.pose).copy() for node in pose_graph.nodes]
    logger.info("Pose-graph optimisation complete")
    return optimized
