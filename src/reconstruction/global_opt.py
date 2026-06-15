"""Global pose graph optimization."""

import logging
from typing import Optional, Tuple

import numpy as np

from reconstruction.types import GlobalOptimizationResult, LoopClosureResult

logger = logging.getLogger(__name__)

# Reciprocal variance for odometry edges (high confidence in local motion).
_ODOMETRY_INFO_SCALE = 100.0
# Fallback information scale for loop edges when no ICP matrix is available.
_LOOP_FALLBACK_INFO_SCALE = 10.0


def build_pose_graph(
    odometry_poses: np.ndarray,
    loop_closures: LoopClosureResult,
) -> "PoseGraph":
    """
    Build a pose graph from odometry and loop closures.

    Each odometry node is initialised from the KISS-ICP / SPOT-aligned pose.
    Consecutive-frame edges carry the relative motion from scan i to scan i+1.
    Loop closure edges carry the ICP-measured relative transform plus a
    data-driven 6×6 information matrix.

    Args:
        odometry_poses: Nx7 or Nx8 poses [optional_t, x, y, z, qx, qy, qz, qw]
        loop_closures: LoopClosureResult from process_loop_closures

    Returns:
        Open3D PoseGraph ready for optimization
    """
    import open3d as o3d

    # Accept both Nx8 (with timestamp) and Nx7 (without).
    if odometry_poses.shape[1] == 8:
        pose_data = odometry_poses[:, 1:]   # strip timestamp → Nx7
    else:
        pose_data = odometry_poses           # already Nx7

    pose_graph = o3d.pipelines.registration.PoseGraph()

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for i in range(len(pose_data)):
        T = transform_7d_to_4x4(pose_data[i])
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T))

    logger.info(f"Added {len(pose_data)} nodes to pose graph")

    # ── Odometry edges ────────────────────────────────────────────────────────
    # The edge transformation must map source-frame points into target-frame
    # coordinates, i.e. T_{(i+1)←i}.
    #
    # With each pose stored as T_world←sensor_i (KISS-ICP convention):
    #   T_{(i+1)←i}  =  inv(T_world←sensor_{i+1}) @ T_world←sensor_i
    #                 =  relative_pose(tgt, src)
    #
    # The previous code called relative_pose(src, tgt) which produced the
    # inverse — T_{i←(i+1)} — causing the optimizer to effectively undo every
    # odometry step and collapse / invert the map.
    info_odom = np.eye(6) * _ODOMETRY_INFO_SCALE
    for i in range(len(pose_data) - 1):
        # T_{(i+1)←i}: maps points from frame i into frame i+1
        rel_7d = relative_pose(pose_data[i + 1], pose_data[i])
        rel_4x4 = transform_7d_to_4x4(rel_7d)

        edge = o3d.pipelines.registration.PoseGraphEdge(
            source_node_id=i,
            target_node_id=i + 1,
            transformation=rel_4x4,
            information=info_odom,
            uncertain=False,
            confidence=1.0,
        )
        pose_graph.edges.append(edge)

    logger.info(f"Added {len(pose_data) - 1} odometry edges")

    # ── Loop closure edges ────────────────────────────────────────────────────
    # ICP returns T such that p_tgt = T @ p_src (source → target).
    # That is already T_{tgt←src} which is what Open3D expects for an edge
    # (source_node_id=src, target_node_id=tgt).  No inversion needed here.
    for (src_idx, tgt_idx), transform_7d in loop_closures.registered_pairs.items():
        transform_4x4 = transform_7d_to_4x4(transform_7d)

        # Use the per-edge information matrix if available; fall back to scalar.
        info = loop_closures.information_matrices.get((src_idx, tgt_idx))
        if info is None:
            info = np.eye(6) * _LOOP_FALLBACK_INFO_SCALE

        edge = o3d.pipelines.registration.PoseGraphEdge(
            source_node_id=int(src_idx),
            target_node_id=int(tgt_idx),
            transformation=transform_4x4,
            information=info,
            uncertain=True,
            confidence=0.8,
        )
        pose_graph.edges.append(edge)

    logger.info(f"Added {len(loop_closures.registered_pairs)} loop closure edges")
    return pose_graph


def optimize_pose_graph(
    pose_graph: "PoseGraph",
    max_iterations: int = 100,
) -> Tuple[np.ndarray, float]:
    """
    Optimize pose graph using Levenberg-Marquardt.

    Args:
        pose_graph: Pose graph to optimize
        max_iterations: Maximum optimization iterations

    Returns:
        Tuple of (optimized_poses Nx7 [x,y,z,qx,qy,qz,qw], final_residual)
    """
    import open3d as o3d

    logger.info(f"Optimizing pose graph ({len(pose_graph.nodes)} nodes, "
                f"{len(pose_graph.edges)} edges)")

    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = max_iterations
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.1,
        edge_prune_threshold=0.25,
        reference_node=0,
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph, method, criteria, option
    )

    optimized_poses = np.array([
        transform_4x4_to_7d(node.pose) for node in pose_graph.nodes
    ])
    logger.info(f"Optimization complete: {len(optimized_poses)} optimized poses")
    return optimized_poses, 0.0


def relative_pose(
    pose1_7d: np.ndarray, pose2_7d: np.ndarray
) -> np.ndarray:
    """
    Compute the relative pose from pose2 to pose1: inv(pose1) * pose2.

    In other words, the returned transform maps a point expressed in pose2's
    frame into pose1's frame.

    Convention: all poses are [x, y, z, qx, qy, qz, qw] (7-element, scalar-last).
    """
    from utils.transforms import invert_transform, compose_transforms

    pos1_inv, quat1_inv = invert_transform(pose1_7d[:3], pose1_7d[3:7])
    relative_pos, relative_quat = compose_transforms(
        pos1_inv, quat1_inv, pose2_7d[:3], pose2_7d[3:7]
    )
    return np.array([
        relative_pos[0], relative_pos[1], relative_pos[2],
        relative_quat[0], relative_quat[1], relative_quat[2], relative_quat[3],
    ])


def transform_7d_to_4x4(pose_7d: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to 4×4 transform matrix."""
    from utils.transforms import quaternion_to_rotation_matrix

    R = quaternion_to_rotation_matrix(pose_7d[3:7])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pose_7d[:3]
    return T


def transform_4x4_to_7d(T: np.ndarray) -> np.ndarray:
    """Convert 4×4 transform matrix to [x, y, z, qx, qy, qz, qw]."""
    from utils.transforms import rotation_matrix_to_quaternion

    pos = T[:3, 3]
    quat = rotation_matrix_to_quaternion(T[:3, :3])
    return np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
