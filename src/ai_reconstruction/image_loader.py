"""Discover and load camera images from a recording session."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from ai_reconstruction.types import ImageRecord

logger = logging.getLogger(__name__)

# Source names written by capture/writer.py
ALL_CAMERA_SOURCES = [
    "back_fisheye_image",
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
]


def list_session_images(
    session_path: Path,
    sources: Optional[List[str]] = None,
) -> List[ImageRecord]:
    """
    Discover all camera images written by DiskWriter and return sorted records.

    Filename convention: images/{frame_id:05d}_{source_name}.png

    Args:
        session_path: Root session directory
        sources: Camera source names to include (None = all found)

    Returns:
        List of ImageRecord sorted by (source_name, frame_id)
    """
    images_dir = session_path / "images"
    if not images_dir.exists():
        logger.warning(f"No images/ directory in {session_path}")
        return []

    records: List[ImageRecord] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in sorted(images_dir.glob(ext)):
            stem = path.stem   # e.g. "00042_frontleft_fisheye_image"
            parts = stem.split("_", 1)
            if len(parts) != 2:
                continue
            try:
                frame_id = int(parts[0])
            except ValueError:
                continue
            source_name = parts[1]
            if sources and source_name not in sources:
                continue
            records.append(ImageRecord(
                path=path,
                source_name=source_name,
                frame_id=frame_id,
            ))

    if not records:
        logger.warning(f"No images found in {images_dir} (sources={sources})")
        return records

    _infer_timestamps(records, session_path)
    records.sort(key=lambda r: (r.source_name, r.frame_id))

    found_sources = set(r.source_name for r in records)
    logger.info(
        f"Found {len(records)} images from {len(found_sources)} cameras: {found_sources}"
    )
    return records


def _infer_timestamps(records: List[ImageRecord], session_path: Path) -> None:
    """
    Populate inferred_timestamp on each record using session start time.

    The writer does not save per-image timestamp sidecars (unlike LiDAR), so
    we reconstruct approximate timestamps as:
        ts ≈ session_start + source_index * (1 / image_sample_rate)

    The default rate of 5 Hz gives 0.2 s spacing, which is good enough for
    matching to SPOT poses in the geometric fallback model.
    """
    start_ts = 0.0
    try:
        from recording.session import load_session
        session = load_session(session_path)
        if session and session.metadata.start_time:
            start_ts = session.metadata.start_time.timestamp()
    except Exception:
        pass

    IMAGE_RATE_HZ = 5.0  # default capture rate from config

    by_source: dict = defaultdict(list)
    for r in records:
        by_source[r.source_name].append(r)

    for source_records in by_source.values():
        source_records.sort(key=lambda r: r.frame_id)
        for idx, r in enumerate(source_records):
            r.inferred_timestamp = start_ts + idx / IMAGE_RATE_HZ
            r.camera_idx = idx


def load_image_rgb(path: Path, target_size: Optional[int] = None) -> np.ndarray:
    """
    Load an image as HxWx3 uint8 RGB, optionally resizing.

    Args:
        path: Image file path
        target_size: Resize so the longer edge = target_size (None = no resize)

    Returns:
        HxWx3 uint8 RGB ndarray
    """
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    # DiskWriter saves images as RGB (BGR→RGB before imwrite), so imread bytes
    # are already [R,G,B] — no cvtColor needed; a second swap would invert them.
    img_rgb = img

    if target_size is not None:
        h, w = img_rgb.shape[:2]
        scale = target_size / max(h, w)
        if scale < 1.0:
            img_rgb = cv2.resize(
                img_rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
    return img_rgb


def load_camera_intrinsics(session_path: Path) -> dict:
    """
    Load camera intrinsics from intrinsics.json.

    Returns:
        Dict: source_name -> {fx, fy, cx, cy, distortion}
        Empty dict if file absent.
    """
    try:
        from recording.session import load_intrinsics
        return load_intrinsics(session_path)
    except Exception as e:
        logger.warning(f"Could not load intrinsics from {session_path}: {e}")
        return {}
