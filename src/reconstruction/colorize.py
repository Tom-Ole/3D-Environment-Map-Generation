"""
Vertex colorization of the SLAM mesh using recorded camera images.

Algorithm
---------
1. Load mesh vertices + normals from the Poisson reconstruction.
2. Load keyframe SLAM poses (world frame, saved during pipeline stage 6)
   and matching LiDAR frame IDs.
3. For every keyframe, find the camera image(s) whose filename frame_id is
   closest to the keyframe's LiDAR frame_id (images have no timestamps on
   disk; frame_id proximity is the best available proxy for time alignment).
4. For each (keyframe, camera) pair, transform every mesh vertex into the
   camera frame, project it with the stored intrinsics, and score the
   projection by viewing angle and distance.
5. Accumulate weighted colours across all views and write a vertex-coloured PLY.

Assumptions / known gaps
------------------------
* Camera frame ≈ body frame ≈ LiDAR origin.  No body-to-camera extrinsic is
  stored in the session; the cameras are offset ~0.3-0.5 m from the body
  centre, which is acceptable for environments > 2 m in extent.
* Images stored on disk already have per-camera rotations applied
  (right=ROTATE_180, frontleft/frontright=ROTATE_90_CLOCKWISE).  The inverse
  rotation is applied on load so that the stored intrinsics (recorded before
  rotation) match the pixel coordinate system used here.
* No ray-cast occlusion: a vertex is assumed visible whenever it projects
  inside the image and has positive depth.  Interior surfaces behind walls may
  receive spurious colours from opposite cameras.
* The SLAM world frame starts at identity for the first scan.  It is NOT the
  same as the SPOT vision frame stored in poses.npy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

# Per-camera rotation applied by image_client.py before saving to disk.
# Values are cv2.ROTATE_* codes; None means no rotation.
_CAMERA_ROTATION_ON_DISK = {
    "back_fisheye_image": None,
    "left_fisheye_image": None,
    "right_fisheye_image": cv2.ROTATE_180,
    "frontleft_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
    "frontright_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
}

# Inverse rotations to undo what image_client applied.
_CAMERA_ROTATION_INVERSE = {
    "back_fisheye_image": None,
    "left_fisheye_image": None,
    "right_fisheye_image": cv2.ROTATE_180,              # self-inverse
    "frontleft_fisheye_image": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "frontright_fisheye_image": cv2.ROTATE_90_COUNTERCLOCKWISE,
}

_DEFAULT_FOV_DEG = 150.0   # conservative FOV for fisheye cameras without intrinsics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def colorize_mesh(
    session_path: Path,
    mesh_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    cameras: Optional[List[str]] = None,
    max_images_per_camera: Optional[int] = None,
    min_view_weight: float = 0.05,
) -> Path:
    """
    Project camera images onto mesh vertices and write a coloured PLY.

    Args:
        session_path:        Root of the recorded session folder.
        mesh_path:           Path to OBJ/PLY mesh.  Defaults to
                             session/reconstruction/mesh.ply.
        output_path:         Destination PLY.  Defaults to
                             session/reconstruction/mesh_colored.ply.
        cameras:             Camera source names to use.  Defaults to all
                             cameras found in images/.
        max_images_per_camera: Cap images per camera (for speed; None = all).
        min_view_weight:     Minimum accumulated weight for a vertex to receive
                             colour (otherwise it is marked grey).

    Returns:
        Path to the written coloured PLY.
    """
    session_path = Path(session_path)
    recon_dir = session_path / "reconstruction"

    mesh_path = mesh_path or recon_dir / "mesh.ply"
    if not mesh_path.exists():
        alt = recon_dir / "mesh.obj"
        if alt.exists():
            mesh_path = alt
        else:
            raise FileNotFoundError(f"No mesh found at {mesh_path} or {alt}")

    output_path = output_path or recon_dir / "mesh_colored.ply"

    logger.info(f"Loading mesh from {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)   # Nv×3
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)  # Nv×3
    nv = len(vertices)
    logger.info(f"Mesh: {nv} vertices, {len(mesh.triangles)} triangles")

    kf_poses, kf_fids = _load_keyframe_data(recon_dir)
    logger.info(f"Loaded {len(kf_poses)} keyframe poses")

    intrinsics = _load_intrinsics(session_path)
    images_dir = session_path / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"No images/ folder in {session_path}")

    image_index = _build_image_index(images_dir, cameras, max_images_per_camera)
    logger.info(
        f"Image index: {sum(len(v) for v in image_index.values())} images "
        f"across {len(image_index)} frame_ids"
    )

    accum_color = np.zeros((nv, 3), dtype=np.float64)
    accum_weight = np.zeros(nv, dtype=np.float64)

    for kf_idx, (T_world_cam, fid) in enumerate(zip(kf_poses, kf_fids)):
        cam_images = _find_images_for_frame(image_index, int(fid), cameras)
        if not cam_images:
            continue

        for source_name, img_path in cam_images:

            img = _load_image(img_path, source_name)
            if img is None:
                continue

            K, dist_coeffs, model = _get_camera_intrinsics(
                intrinsics, source_name, img.shape
            )

            T_cam_world = np.linalg.inv(T_world_cam)
            R = T_cam_world[:3, :3]
            t = T_cam_world[:3, 3]

            # Transform all vertices into this camera frame
            pts_cam = (R @ vertices.T).T + t  # Nv×3

            # Only vertices in front of the camera
            depth = pts_cam[:, 2]
            front_mask = depth > 0.1

            if not np.any(front_mask):
                continue

            pts_front = pts_cam[front_mask]          # Nf×3
            norm_front = normals[front_mask]          # Nf×3

            pts_2d = _project(pts_front, K, dist_coeffs, model)  # Nf×2

            h, w = img.shape[:2]
            in_bounds = (
                (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w - 1) &
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h - 1)
            )

            if not np.any(in_bounds):
                continue

            vi_front = np.where(front_mask)[0]
            vi_valid = vi_front[in_bounds]
            pts_valid = pts_front[in_bounds]
            norm_valid = norm_front[in_bounds]
            pts_2d_valid = pts_2d[in_bounds]

            # Viewing direction (pointing toward camera)
            dist = np.linalg.norm(pts_valid, axis=1) + 1e-6
            view_dir = -pts_valid / dist[:, None]

            # Angle between surface normal and view direction (0-1)
            dot = np.einsum("ij,ij->i", norm_valid, view_dir)
            angle_w = np.maximum(dot, 0.0)

            # Distance weight (closer = better)
            dist_w = 1.0 / (dist + 1.0)

            weights = angle_w * dist_w  # Nvalid

            # Sample image colours (bilinear, float 0-1)
            colors = _sample_bilinear(img, pts_2d_valid)  # Nvalid×3

            accum_color[vi_valid] += weights[:, None] * colors
            accum_weight[vi_valid] += weights

        if (kf_idx + 1) % 10 == 0:
            logger.info(
                f"  Processed {kf_idx + 1}/{len(kf_poses)} keyframes …"
            )

    # Normalise
    valid = accum_weight >= min_view_weight
    final_colors = np.full((nv, 3), 0.5, dtype=np.float64)
    final_colors[valid] = accum_color[valid] / accum_weight[valid, None]
    final_colors = np.clip(final_colors, 0.0, 1.0)

    colored = sum(valid)
    logger.info(
        f"Coloured {colored}/{nv} vertices "
        f"({100.0 * colored / nv:.1f} %)"
    )

    mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    logger.info(f"Saved coloured mesh → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_keyframe_data(recon_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load keyframe_poses.npy (Mx4x4) and keyframe_frame_ids.npy (M,)."""
    poses_path = recon_dir / "keyframe_poses.npy"
    fids_path = recon_dir / "keyframe_frame_ids.npy"

    if not poses_path.exists():
        raise FileNotFoundError(
            f"{poses_path} not found.  Re-run the reconstruction pipeline "
            "to generate keyframe poses (requires updated pipeline.py)."
        )
    poses = np.load(str(poses_path))      # Mx4x4
    fids = np.load(str(fids_path))        # M,
    return poses, fids


def _load_intrinsics(session_path: Path) -> dict:
    p = session_path / "intrinsics.json"
    if not p.exists():
        logger.warning("intrinsics.json not found — using default pinhole model")
        return {}
    with open(p) as f:
        return json.load(f)


def _get_camera_intrinsics(
    intrinsics: dict, source_name: str, img_shape: Tuple
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Return (K 3×3, dist_coeffs or None, model_name)."""
    h, w = img_shape[:2]
    data = intrinsics.get(source_name, {})

    if data:
        fx = float(data.get("fx", w / 2))
        fy = float(data.get("fy", h / 2))
        cx = float(data.get("cx", w / 2))
        cy = float(data.get("cy", h / 2))
        dist = data.get("distortion", {})
        model = dist.get("model", "pinhole")
    else:
        # Default: focal length = half image width (rough fisheye estimate)
        fx = fy = w / 2.0
        cx, cy = w / 2.0, h / 2.0
        dist = {}
        model = "pinhole"

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    coeffs = None
    if model == "kannala_brandt" and all(k in dist for k in ("k1", "k2", "k3", "k4")):
        coeffs = np.array([dist["k1"], dist["k2"], dist["k3"], dist["k4"]])
        model = "kannala_brandt"
    else:
        model = "pinhole"

    return K, coeffs, model


# ---------------------------------------------------------------------------
# Image discovery and loading
# ---------------------------------------------------------------------------

def _build_image_index(
    images_dir: Path,
    cameras: Optional[List[str]],
    max_images_per_camera: Optional[int] = None,
) -> Dict[int, List[Tuple[str, Path]]]:
    """Return {frame_id: [(source_name, path), ...]} for all images.

    When max_images_per_camera is set, only the first N frame_ids for each
    camera source are included, uniformly spaced across the full timeline.
    """
    # Collect per-camera sorted lists first so we can apply the cap
    per_camera: Dict[str, List[Tuple[int, Path]]] = {}
    for img_path in sorted(images_dir.glob("*.png")):
        stem = img_path.stem
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            fid = int(parts[0])
        except ValueError:
            continue
        src = parts[1]
        if cameras and src not in cameras:
            continue
        per_camera.setdefault(src, []).append((fid, img_path))

    index: Dict[int, List[Tuple[str, Path]]] = {}
    for src, entries in per_camera.items():
        entries.sort(key=lambda x: x[0])
        if max_images_per_camera is not None and len(entries) > max_images_per_camera:
            # Uniformly sample across timeline instead of just taking the first N
            step = len(entries) / max_images_per_camera
            entries = [entries[int(i * step)] for i in range(max_images_per_camera)]
        for fid, img_path in entries:
            index.setdefault(fid, []).append((src, img_path))
    return index


def _find_images_for_frame(
    index: Dict[int, List[Tuple[str, Path]]], frame_id: int,
    cameras: Optional[List[str]]
) -> List[Tuple[str, Path]]:
    """Find images for the closest available frame_id."""
    if not index:
        return []
    if frame_id in index:
        return index[frame_id]
    # Nearest available frame_id
    all_fids = np.array(sorted(index.keys()))
    nearest = int(all_fids[np.argmin(np.abs(all_fids - frame_id))])
    return index[nearest]


def _load_image(path: Path, source_name: str) -> Optional[np.ndarray]:
    """Load image as float32 RGB (0-1), undoing the on-disk rotation.

    DiskWriter saves images by converting BGR → RGB before cv2.imwrite, so the
    bytes on disk are already in R-G-B order.  cv2.imread reads those bytes into
    its array without reordering, meaning array[y,x] = [R, G, B] — which is
    already correct RGB despite OpenCV's BGR convention.  A second BGR↔RGB swap
    here would invert the channels, so we deliberately skip it.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning(f"Could not read {path}")
        return None
    # Do NOT cvtColor(BGR2RGB): the file was saved as RGB by DiskWriter;
    # imread gives us the correct [R,G,B] bytes already.

    inv_rot = _CAMERA_ROTATION_INVERSE.get(source_name)
    if inv_rot is not None:
        img = cv2.rotate(img, inv_rot)

    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _project(
    pts_cam: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    model: str,
) -> np.ndarray:
    """Project Nx3 camera-frame points → Nx2 pixel coordinates."""
    if model == "kannala_brandt" and dist_coeffs is not None:
        return _project_kannala_brandt(pts_cam, K, dist_coeffs)
    return _project_pinhole(pts_cam, K)


def _project_pinhole(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Standard pinhole projection (no distortion)."""
    z = pts_cam[:, 2:3]
    xy = pts_cam[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * xy[:, 0] + K[0, 2]
    v = K[1, 1] * xy[:, 1] + K[1, 2]
    return np.stack([u, v], axis=1)


def _project_kannala_brandt(
    pts_cam: np.ndarray, K: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """
    Kannala-Brandt equidistant fisheye projection.

    r(θ) = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹
    """
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(rho, Z)
    theta2 = theta * theta
    r = theta * (1.0 + theta2 * (k[0] + theta2 * (k[1] + theta2 * (k[2] + theta2 * k[3]))))

    # Avoid division by zero for points on the optical axis
    safe_rho = np.where(rho < 1e-9, 1e-9, rho)
    mx = r * X / safe_rho
    my = r * Y / safe_rho

    u = K[0, 0] * mx + K[0, 2]
    v = K[1, 1] * my + K[1, 2]
    return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Colour sampling
# ---------------------------------------------------------------------------

def _sample_bilinear(img: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
    """
    Bilinear colour sampling.

    Args:
        img:    H×W×3 float32 image (0-1).
        pts_2d: N×2 float pixel coordinates.

    Returns:
        N×3 float32 RGB colours.
    """
    h, w = img.shape[:2]
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    dx = (x - np.floor(x)).astype(np.float32)
    dy = (y - np.floor(y)).astype(np.float32)

    c00 = img[y0, x0]  # top-left
    c01 = img[y1, x0]  # bottom-left
    c10 = img[y0, x1]  # top-right
    c11 = img[y1, x1]  # bottom-right

    w00 = ((1.0 - dx) * (1.0 - dy))[:, None]
    w01 = ((1.0 - dx) * dy)[:, None]
    w10 = (dx * (1.0 - dy))[:, None]
    w11 = (dx * dy)[:, None]

    return w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11
