"""Fisheye -> pinhole undistortion for SPOT Kannala-Brandt cameras.

MASt3R / DUSt3R assume a roughly pinhole camera and fit a single focal length.
SPOT's fisheye lenses are strongly distorted, which bows straight edges and
breaks multi-view consistency, so we rectify each image to a pinhole model
before reconstruction.

SPOT images are saved upright (rotated at capture time, see
capture/image_client.py CAMERA_ROTATION) but the stored intrinsics correspond
to the original *sensor* orientation. So we un-rotate each image back to sensor
frame, undistort with the stored Kannala-Brandt parameters, then re-apply the
capture-time rotation so the rectified output stays upright.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _rotation_maps():
    """cv2 rotation codes to undo / redo the capture-time rotation per source."""
    import cv2

    # Inverse of capture/image_client.py CAMERA_ROTATION (sources not listed
    # were saved without rotation).
    unrotate = {
        "frontleft_fisheye_image": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "frontright_fisheye_image": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "right_fisheye_image": cv2.ROTATE_180,
    }
    rerotate = {
        "frontleft_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
        "frontright_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
        "right_fisheye_image": cv2.ROTATE_180,
    }
    return unrotate, rerotate


def _kb_params(entry: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (K, D) for a Kannala-Brandt camera, or None if not applicable."""
    dist = entry.get("distortion") or {}
    if dist.get("model") != "kannala_brandt":
        return None
    fx, fy = entry.get("fx"), entry.get("fy")
    cx, cy = entry.get("cx"), entry.get("cy")
    if not fx or not fy:
        return None
    K = np.array([[fx, 0.0, cx or 0.0],
                  [0.0, fy, cy or 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([dist.get("k1", 0.0), dist.get("k2", 0.0),
                  dist.get("k3", 0.0), dist.get("k4", 0.0)], dtype=np.float64)
    return K, D


def has_fisheye(source_names: List[str], intrinsics: Dict[str, dict]) -> bool:
    """True if any source has usable Kannala-Brandt intrinsics."""
    return any(_kb_params(intrinsics.get(s, {})) is not None for s in source_names)


def undistort_images(
    image_paths: List[Path],
    source_names: List[str],
    intrinsics: Dict[str, dict],
    out_dir: Path,
    balance: float = 0.0,
) -> List[Path]:
    """Rectify fisheye images to a pinhole model and write them to out_dir.

    Args:
        image_paths:  saved (upright) image paths.
        source_names: camera source per image (e.g. "frontleft_fisheye_image").
        intrinsics:   source_name -> {fx, fy, cx, cy, distortion}.
        out_dir:      directory to write rectified images into.
        balance:      cv2.fisheye balance for the new camera matrix; 0 crops to
                      the valid (black-border-free) region, 1 keeps full FOV.

    Returns output paths aligned with image_paths. Images without a usable
    Kannala-Brandt model are copied through unchanged.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    unrotate, rerotate = _rotation_maps()
    # Rectification maps are expensive; cache per (source, h, w).
    maps: Dict[tuple, tuple] = {}
    out_paths: List[Path] = []
    n_rectified = 0

    for path, source in zip(image_paths, source_names):
        out_path = out_dir / path.name
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"undistort: could not read {path}; passing through")
            out_paths.append(path)
            continue

        kb = _kb_params(intrinsics.get(source, {}))
        if kb is None:
            cv2.imwrite(str(out_path), img)
            out_paths.append(out_path)
            continue
        K, D = kb

        # Back to sensor orientation so K / D align with the pixel grid.
        un = unrotate.get(source)
        sensor = cv2.rotate(img, un) if un is not None else img
        h, w = sensor.shape[:2]

        key = (source, h, w)
        if key not in maps:
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=balance)
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_k, (w, h), cv2.CV_16SC2)
            maps[key] = (m1, m2)
        m1, m2 = maps[key]

        rect = cv2.remap(sensor, m1, m2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        re = rerotate.get(source)
        if re is not None:
            rect = cv2.rotate(rect, re)

        cv2.imwrite(str(out_path), rect)
        out_paths.append(out_path)
        n_rectified += 1

    logger.info(
        f"undistort: rectified {n_rectified}/{len(image_paths)} "
        f"fisheye images -> {out_dir}"
    )
    return out_paths
