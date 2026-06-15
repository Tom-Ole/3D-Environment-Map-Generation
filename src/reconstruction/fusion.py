"""
Point cloud fusion and Poisson surface reconstruction.

Fusion:
  Transform each keyframe scan by its optimised world pose, accumulate
  into a single PointCloud, and voxel-downsample periodically to keep
  memory bounded.

Meshing:
  Estimate normals on the fused cloud, run Open3D's screened Poisson
  reconstruction, and prune low-density boundary artefacts.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

_POISSON_DEPTH = 9
_DENSITY_PRUNE_PCT = 0.05       # remove the bottom 5 % of density vertices
_NORMAL_RADIUS = 0.3            # metres for KD-tree normal estimation
_NORMAL_MAX_NN = 30
_ORIENT_K = 15                  # neighbours for consistent normal orientation


def fuse_point_clouds(
    scan_paths: List[Path],
    poses: List[np.ndarray],
    voxel_size: float = 0.05,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> o3d.geometry.PointCloud:
    """
    Accumulate transformed scans into a single downsampled map.

    Args:
        scan_paths: PLY paths (one per keyframe).
        poses: Corresponding 4x4 world-frame SE(3) matrices.
        voxel_size: Output voxel grid resolution (m).
        progress_cb: Called with (done, total).

    Returns:
        Fused Open3D PointCloud.
    """
    accumulated = o3d.geometry.PointCloud()
    n = len(scan_paths)
    downsample_every = max(1, min(50, n // 10))  # partial downsample cadence

    for i, (path, T) in enumerate(zip(scan_paths, poses)):
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            continue

        pcd.transform(T)
        accumulated += pcd

        # Periodically downsample to bound memory during accumulation
        if (i + 1) % downsample_every == 0 and voxel_size > 0:
            accumulated = accumulated.voxel_down_sample(voxel_size)

        if progress_cb:
            progress_cb(i + 1, n)

    # Final downsample
    if voxel_size > 0 and len(accumulated.points) > 0:
        accumulated = accumulated.voxel_down_sample(voxel_size)

    logger.info(f"Fused map: {len(accumulated.points)} points from {n} keyframes")
    return accumulated


def reconstruct_mesh(
    cloud: o3d.geometry.PointCloud,
    depth: int = _POISSON_DEPTH,
) -> Optional[o3d.geometry.TriangleMesh]:
    """
    Poisson surface reconstruction on a dense point cloud.

    Args:
        cloud: Fused input cloud (normals estimated here).
        depth: Poisson octree depth – higher → finer mesh, slower.

    Returns:
        Cleaned triangle mesh, or None if reconstruction fails.
    """
    if len(cloud.points) < 100:
        logger.warning(f"Too few points for meshing ({len(cloud.points)})")
        return None

    # Estimate oriented normals
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=_NORMAL_RADIUS, max_nn=_NORMAL_MAX_NN
        )
    )
    cloud.orient_normals_consistent_tangent_plane(k=_ORIENT_K)

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=depth
        )
    except Exception as e:
        logger.error(f"Poisson reconstruction failed: {e}")
        return None

    # Remove low-density artefacts (open boundaries / noise)
    densities_np = np.asarray(densities)
    threshold = np.quantile(densities_np, _DENSITY_PRUNE_PCT)
    vertices_to_remove = (densities_np < threshold).tolist()
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    logger.info(
        f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
    )
    return mesh


def save_results(
    cloud: o3d.geometry.PointCloud,
    mesh: Optional[o3d.geometry.TriangleMesh],
    output_dir: Path,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Write cloud and mesh to output_dir/reconstruction/.

    Returns:
        (cloud_path, mesh_ply_path, mesh_obj_path) — mesh paths are None if
        no mesh was produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cloud_path = output_dir / "cloud_optimized.ply"
    o3d.io.write_point_cloud(str(cloud_path), cloud)
    logger.info(f"Cloud saved → {cloud_path} ({len(cloud.points)} pts)")

    mesh_ply_path: Optional[Path] = None
    mesh_obj_path: Optional[Path] = None

    if mesh is not None:
        mesh_ply_path = output_dir / "mesh.ply"
        mesh_obj_path = output_dir / "mesh.obj"
        o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)
        o3d.io.write_triangle_mesh(str(mesh_obj_path), mesh)
        logger.info(f"Mesh saved → {mesh_ply_path}")

    return cloud_path, mesh_ply_path, mesh_obj_path
