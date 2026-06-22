"""Export AI reconstruction results to session/ai_reconstruction/."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ai_reconstruction.types import (
    AIPointCloudResult,
    AIReconstructionConfig,
    AIReconstructionResult,
)

logger = logging.getLogger(__name__)


def export_results(
    result: AIPointCloudResult,
    config: AIReconstructionConfig,
    session_path: Path,
    duration: float,
    keyframe_count: int,
) -> AIReconstructionResult:
    """
    Persist all AI reconstruction outputs under session_path/ai_reconstruction/.

    Outputs:
        point_cloud.ply   — colored PLY point cloud
        camera_poses.npy  — Mx4x4 float64 camera-to-world matrices
        metadata.json     — run configuration and statistics

    Args:
        result:         Raw AIPointCloudResult from the model
        config:         Pipeline configuration used for this run
        session_path:   Root session directory
        duration:       Wall-clock pipeline duration (seconds)
        keyframe_count: Number of keyframes processed

    Returns:
        AIReconstructionResult with paths and statistics
    """
    out_dir = session_path / "ai_reconstruction"
    out_dir.mkdir(parents=True, exist_ok=True)

    final = AIReconstructionResult(
        model_used=result.model_name,
        point_count=int(len(result.points)) if result.points is not None else 0,
        keyframe_count=keyframe_count,
        image_count=len(result.image_paths) if result.image_paths else 0,
        device_used=config.device.value,
        duration_seconds=round(duration, 2),
    )

    # ── Point cloud ───────────────────────────────────────────────────────────
    if result.points is not None and len(result.points) > 0:
        pcd_path = out_dir / "point_cloud.ply"
        _save_point_cloud(result.points, result.colors, pcd_path)
        final.point_cloud_path = pcd_path

    # ── Surface mesh (Poisson) ────────────────────────────────────────────────
    if config.mesh_enabled and result.points is not None and len(result.points) >= 100:
        mesh_path = out_dir / "mesh.ply"
        if _save_mesh_poisson(
            result.points, result.colors, mesh_path,
            depth=config.poisson_depth,
            density_quantile=config.mesh_density_quantile,
            normal_radius=config.normal_radius,
        ) is not None:
            final.mesh_path = mesh_path

    # ── Camera poses ──────────────────────────────────────────────────────────
    if result.camera_poses is not None and len(result.camera_poses) > 0:
        poses_path = out_dir / "camera_poses.npy"
        np.save(str(poses_path), result.camera_poses.astype(np.float64))
        final.camera_poses_path = poses_path
        logger.info(f"Saved {len(result.camera_poses)} camera poses to {poses_path}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "model": result.model_name,
        "metric_scale": result.metric_scale,
        "point_count": final.point_count,
        "mesh_exported": final.mesh_path is not None,
        "keyframe_count": keyframe_count,
        "image_count": final.image_count,
        "device": config.device.value,
        "image_size": config.image_size,
        "camera_sources": config.camera_sources,
        "keyframe_strategy": config.keyframe_strategy.value,
        "keyframe_interval": config.keyframe_interval,
        "max_images": config.max_images,
        "voxel_size": config.voxel_size,
        "global_alignment_iter": config.global_alignment_iter,
        "duration_seconds": final.duration_seconds,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    final.metadata_path = meta_path

    final.success = True
    logger.info(
        f"AI reconstruction exported to {out_dir}  "
        f"({final.point_count} pts | {duration:.1f} s)"
    )
    return final


# ── Private helpers ───────────────────────────────────────────────────────────

def _save_mesh_poisson(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    path: Path,
    depth: int = 10,
    density_quantile: float = 0.02,
    normal_radius: float = 0.1,
) -> Optional[Path]:
    """Build a surface mesh via Poisson reconstruction and save it.

    Poisson needs oriented normals, so we estimate and consistently orient them
    first. It produces a watertight surface that extends beyond the observed
    samples, so we trim the lowest-density (least-supported) vertices to drop
    hallucinated geometry. normal_radius assumes metric-scale points (metres).

    Returns the path on success, or None if meshing was skipped/failed.
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("Open3D not available — skipping mesh export")
        return None

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=30)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
        densities = np.asarray(densities)
        if density_quantile > 0 and len(densities) > 0:
            thr = np.quantile(densities, density_quantile)
            mesh.remove_vertices_by_mask(densities < thr)

        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(path), mesh)
        logger.info(
            f"Saved Poisson mesh ({len(mesh.vertices)} verts, "
            f"{len(mesh.triangles)} tris) -> {path}"
        )
        return path
    except Exception as e:
        logger.warning(f"Poisson meshing failed: {e}")
        return None


def _save_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    path: Path,
) -> None:
    """Save Nx3 float points + optional Nx3 uint8 colors as binary PLY."""
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
        logger.info(f"Saved PLY cloud ({len(points)} pts) -> {path}")

    except ImportError:
        _write_ascii_ply(points, colors, path)


def _write_ascii_ply(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    path: Path,
) -> None:
    """Minimal ASCII PLY writer used when Open3D is unavailable."""
    n = len(points)
    has_color = colors is not None and len(colors) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            row = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if has_color:
                row += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
            f.write(row + "\n")
    logger.info(f"Saved ASCII PLY ({n} pts) -> {path}")
