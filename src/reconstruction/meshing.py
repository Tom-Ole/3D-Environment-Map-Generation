"""Mesh generation from point clouds (Poisson reconstruction)."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from reconstruction.types import MeshOutput

logger = logging.getLogger(__name__)


def generate_mesh(
    cloud: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.05,
    depth: int = 8,
    scan_origins: Optional[np.ndarray] = None,
) -> Tuple[MeshOutput, object]:
    """
    Generate a Poisson surface mesh from a fused point cloud.

    Normal orientation is a critical parameter for Poisson reconstruction.
    The previous implementation used orient_normals_consistent_tangent_plane
    which propagates orientation via a minimum spanning tree and cannot
    guarantee outward-facing normals for indoor geometry.

    This version uses the world-frame positions of the LiDAR sensor at each
    scan (scan_origins) to orient normals: for each surface point, normals are
    oriented toward the weighted centroid of all scan positions that could
    plausibly have seen it.  This matches what a viewer-directed orientation
    would produce for a range sensor sweeping a room.

    Args:
        cloud: Mx3 fused point cloud in world frame
        colors: Mx3 uint8 RGB colors (optional)
        voxel_size: Controls normal estimation radius and density filter
        depth: Poisson octree depth (higher = more geometric detail)
        scan_origins: Kx3 world-frame positions of the LiDAR sensor per scan.
                      Used to compute the viewpoint for normal orientation.

    Returns:
        Tuple of (MeshOutput statistics, Open3D TriangleMesh)
    """
    import open3d as o3d

    logger.info(
        f"Generating mesh from {len(cloud)} points "
        f"(depth={depth}, voxel_size={voxel_size})"
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

    # ── Statistical outlier removal ───────────────────────────────────────────
    # Remove isolated noise points, glass reflections, and dynamic-object
    # remnants before feeding into Poisson.  With nb_neighbors=20 and
    # std_ratio=2.0 this keeps points that have at least 20 neighbours within
    # 2 standard deviations of the mean neighbour distance.
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    logger.info(f"After outlier removal: {len(pcd.points)} points")

    # ── Normal estimation ─────────────────────────────────────────────────────
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )

    # ── Normal orientation ────────────────────────────────────────────────────
    # Choose a viewpoint that represents "where the sensor was".
    # If per-scan origins are available use their centroid; otherwise fall back
    # to the point cloud centroid (usually inside the scanned space for indoor
    # environments — good enough for the tangent-plane propagation to work
    # consistently from an interior seed).
    if scan_origins is not None and len(scan_origins) > 0:
        viewpoint = np.mean(scan_origins, axis=0)
        logger.info(f"Orienting normals toward sensor centroid {viewpoint}")
    else:
        pts = np.asarray(pcd.points)
        viewpoint = pts.mean(axis=0)
        logger.info("No scan origins provided; orienting normals toward cloud centroid")

    pcd.orient_normals_towards_camera_location(viewpoint)

    # ── Poisson reconstruction ────────────────────────────────────────────────
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Remove low-density vertices (boundary artefacts from Poisson padding).
    densities = np.asarray(densities)
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


def save_mesh(
    mesh,
    output_path: str,
    format: str = "ply",
) -> bool:
    """Save an Open3D TriangleMesh to disk."""
    try:
        import open3d as o3d

        o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
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
    """Save a colored Nx3 point cloud to a PLY file."""
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
