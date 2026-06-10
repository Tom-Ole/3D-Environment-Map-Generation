"""Global pose graph optimization."""

import logging
from typing import Optional, Tuple

import numpy as np

from reconstruction.types import GlobalOptimizationResult, LoopClosureResult

logger = logging.getLogger(__name__)


def build_pose_graph(
    odometry_poses: np.ndarray,
    loop_closures: LoopClosureResult,
    odometry_cov_scale: float = 0.01,
    loop_cov_scale: float = 0.1,
) -> "PoseGraph":
    """
    Build a pose graph from odometry and loop closures.

    Args:
        odometry_poses: Nx7 odometry poses [t, x, y, z, qx, qy, qz, qw]
        loop_closures: LoopClosureResult with registered pairs
        odometry_cov_scale: Covariance scale for odometry edges
        loop_cov_scale: Covariance scale for loop edges

    Returns:
        PoseGraph object ready for optimization
    """
    try:
        import open3d as o3d

        # Create empty pose graph
        pose_graph = o3d.pipelines.registration.PoseGraph()

        # Add nodes (one per pose) — odometry_poses are 8-element [t, x, y, z, qx, qy, qz, qw]
        for i in range(len(odometry_poses)):
            pose_7d = odometry_poses[i]
            pose_4x4 = transform_7d_to_4x4(pose_7d[1:])  # strip timestamp
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose_4x4))

        logger.info(f"Added {len(odometry_poses)} nodes to pose graph")

        # Add odometry edges (consecutive poses)
        for i in range(len(odometry_poses) - 1):
            src_pose = odometry_poses[i]
            tgt_pose = odometry_poses[i + 1]

            # Relative transform — strip timestamp for 7-element helpers
            relative_transform = relative_pose(src_pose[1:], tgt_pose[1:])
            relative_4x4 = transform_7d_to_4x4(relative_transform)

            # Information matrix (inverse covariance)
            information = np.eye(6) / odometry_cov_scale

            edge = o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=i,
                target_node_id=i + 1,
                transformation=relative_4x4,
                information=information,
                uncertain=False,
                confidence=1.0,
            )
            pose_graph.edges.append(edge)

        logger.info(f"Added {len(odometry_poses) - 1} odometry edges")

        # Add loop closure edges
        for (src_idx, tgt_idx), transform_7d in loop_closures.registered_pairs.items():
            transform_4x4 = transform_7d_to_4x4(transform_7d)

            # Information matrix for loop edges (less certain)
            information = np.eye(6) / loop_cov_scale

            edge = o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=int(src_idx),
                target_node_id=int(tgt_idx),
                transformation=transform_4x4,
                information=information,
                uncertain=True,
                confidence=0.8,
            )
            pose_graph.edges.append(edge)

        logger.info(f"Added {len(loop_closures.registered_pairs)} loop closure edges")

        return pose_graph

    except Exception as e:
        logger.error(f"Failed to build pose graph: {e}")
        raise


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
        Tuple of (optimized_poses Nx7, final_residual)
    """
    try:
        import open3d as o3d

        logger.info(f"Optimizing pose graph ({len(pose_graph.nodes)} nodes)")

        # Run optimization (modifies pose_graph in-place, returns None)
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

        # Extract optimized poses from updated nodes
        optimized_poses = []
        for node in pose_graph.nodes:
            pose_7d = transform_4x4_to_7d(node.pose)
            optimized_poses.append(pose_7d)

        optimized_poses = np.array(optimized_poses)
        logger.info(f"Optimization complete: {len(optimized_poses)} optimized poses")

        return optimized_poses, 0.0

    except Exception as e:
        logger.error(f"Failed to optimize pose graph: {e}")
        raise


def relative_pose(
    pose1_7d: np.ndarray, pose2_7d: np.ndarray
) -> np.ndarray:
    """
    Compute relative pose from pose1 to pose2.

    Args:
        pose1_7d: [x, y, z, qx, qy, qz, qw]
        pose2_7d: [x, y, z, qx, qy, qz, qw]

    Returns:
        Relative pose [x, y, z, qx, qy, qz, qw]
    """
    from utils.transforms import invert_transform, compose_transforms

    # Invert pose1: get world-to-pose1
    pos1_inv, quat1_inv = invert_transform(pose1_7d[:3], pose1_7d[3:7])

    # Compose: (world-to-pose1) * (world-to-pose2)
    relative_pos, relative_quat = compose_transforms(
        pos1_inv, quat1_inv, pose2_7d[:3], pose2_7d[3:7]
    )

    return np.array(
        [
            relative_pos[0],
            relative_pos[1],
            relative_pos[2],
            relative_quat[0],
            relative_quat[1],
            relative_quat[2],
            relative_quat[3],
        ]
    )


def transform_7d_to_4x4(pose_7d: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to 4x4 transform matrix."""
    from utils.transforms import quaternion_to_rotation_matrix

    pos = pose_7d[:3]
    quat = pose_7d[3:7]

    R = quaternion_to_rotation_matrix(quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    return T


def transform_4x4_to_7d(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 transform matrix to [x, y, z, qx, qy, qz, qw]."""
    from utils.transforms import rotation_matrix_to_quaternion

    pos = T[:3, 3]
    R = T[:3, :3]
    quat = rotation_matrix_to_quaternion(R)

    return np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
