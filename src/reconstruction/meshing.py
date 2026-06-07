"""Mesh generation from point clouds (TSDF + Poisson)."""

import logging
from typing import Optional, Tuple

import numpy as np

from reconstruction.types import MeshOutput

logger = logging.getLogger(__name__)


def generate_mesh(
    cloud: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.05,
    depth: int = 8,
) -> MeshOutput:
    """
    Generate a mesh from a point cloud using Poisson reconstruction.

    Args:
        cloud: Mx3 point cloud
        colors: Mx3 optional color data
        voxel_size: Voxel size for implicit function
        depth: Octree depth (higher = more detail)

    Returns:
        MeshOutput with generated mesh paths and statistics
    """
    try:
        import open3d as o3d

        logger.info(
            f"Generating mesh from {len(cloud)} points (depth={depth}, voxel_size={voxel_size})"
        )

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

        # Estimate normals (required for Poisson)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        logger.info(
            f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
        )

        return MeshOutput(
            cloud_path="",
            mesh_ply_path="",
            mesh_obj_path="",
            mesh_vertex_count=len(mesh.vertices),
            mesh_face_count=len(mesh.triangles),
        ), mesh

    except Exception as e:
        logger.error(f"Failed to generate mesh: {e}")
        raise


def tsdf_fusion(
    scans: list,
    optimized_poses: np.ndarray,
    voxel_size: float = 0.05,
    sdf_trunc: float = 0.1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fuse point clouds using TSDF (truncated signed distance function).

    Args:
        scans: List of point clouds
        optimized_poses: Nx7 optimized poses
        voxel_size: Voxel size
        sdf_trunc: Truncation parameter

    Returns:
        Tuple of (mesh, cloud) or (None, cloud) if meshing fails
    """
    logger.warning("TSDF fusion not fully implemented, using Poisson only")

    # For now, just fuse and apply Poisson
    from reconstruction.fusion import fuse_and_downsample

    fused = fuse_and_downsample(scans, optimized_poses, voxel_size)
    return None, fused


def save_mesh(
    mesh,
    output_path: str,
    format: str = "ply",
) -> bool:
    """
    Save mesh to file.

    Args:
        mesh: Open3D TriangleMesh object
        output_path: Output file path
        format: "ply" or "obj"

    Returns:
        True if successful, False otherwise
    """
    try:
        import open3d as o3d

        if format == "ply":
            o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
        elif format == "obj":
            o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
        else:
            logger.error(f"Unsupported format: {format}")
            return False

        logger.info(f"Saved mesh to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save mesh: {e}")
        return False


def save_colored_cloud(
    cloud: np.ndarray,
    colors: np.ndarray,
    output_path: str,
) -> bool:
    """
    Save colored point cloud to PLY file.

    Args:
        cloud: Mx3 point cloud
        colors: Mx3 RGB colors
        output_path: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

        o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
        logger.info(f"Saved colored cloud to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save colored cloud: {e}")
        return False
