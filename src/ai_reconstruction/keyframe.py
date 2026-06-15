"""Keyframe selection strategies for the AI reconstruction pipeline."""

import logging
from typing import List, Optional

import numpy as np

from ai_reconstruction.types import ImageRecord, KeyframeStrategy

logger = logging.getLogger(__name__)


def select_keyframes(
    records: List[ImageRecord],
    strategy: KeyframeStrategy = KeyframeStrategy.INTERVAL,
    max_frames: int = 100,
    interval: int = 5,
    spot_poses: Optional[np.ndarray] = None,
    min_translation: float = 0.3,
    min_rotation_deg: float = 10.0,
) -> List[ImageRecord]:
    """
    Return a subset of images to feed into the AI model.

    INTERVAL strategy: take every `interval`-th image — simple, always works.
    MOTION strategy:   select a new keyframe whenever the robot has moved
                       >= min_translation metres or rotated >= min_rotation_deg
                       degrees since the last keyframe.  Falls back to INTERVAL
                       when no SPOT poses are available.

    The hard cap `max_frames` is enforced after strategy selection by
    uniform sub-sampling so that no information is completely lost.

    Args:
        records:           All available image records sorted by (source, frame_id)
        strategy:          INTERVAL or MOTION
        max_frames:        Upper limit on returned keyframes
        interval:          Frame step for INTERVAL strategy
        spot_poses:        Nx8 [ts, x, y, z, qx, qy, qz, qw] for MOTION
        min_translation:   Minimum robot translation between keyframes (m)
        min_rotation_deg:  Minimum robot rotation between keyframes (deg)

    Returns:
        Filtered list of ImageRecord
    """
    if not records:
        return []

    if strategy == KeyframeStrategy.MOTION:
        if spot_poses is not None and len(spot_poses) > 0:
            selected = _motion_based(records, spot_poses, min_translation, min_rotation_deg)
        else:
            logger.warning("Motion-based keyframe selection: no SPOT poses available, using interval.")
            selected = _interval_based(records, interval)
    else:
        selected = _interval_based(records, interval)

    # Enforce hard cap via uniform sub-sampling
    if len(selected) > max_frames:
        step = len(selected) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        selected = [selected[i] for i in indices]

    logger.info(
        f"Keyframe selection [{strategy.value}]: "
        f"{len(selected)} / {len(records)} images"
    )
    return selected


# ── Strategy implementations ──────────────────────────────────────────────────

def _interval_based(records: List[ImageRecord], interval: int) -> List[ImageRecord]:
    return records[::max(1, interval)]


def _motion_based(
    records: List[ImageRecord],
    spot_poses: np.ndarray,
    min_translation: float,
    min_rotation_deg: float,
) -> List[ImageRecord]:
    """Select keyframes whenever the robot has moved sufficiently."""
    from utils.timestamps import interpolate_pose_to_timestamp

    pose_ts = spot_poses[:, 0]
    positions = spot_poses[:, 1:4]
    quaternions = spot_poses[:, 4:8]

    selected: List[ImageRecord] = []
    last_pos: Optional[np.ndarray] = None
    last_quat: Optional[np.ndarray] = None

    for record in records:
        pos, quat = interpolate_pose_to_timestamp(
            record.inferred_timestamp, pose_ts, positions, quaternions
        )
        if pos is None:
            idx = int(np.argmin(np.abs(pose_ts - record.inferred_timestamp)))
            pos = positions[idx].copy()
            quat = quaternions[idx].copy()

        if last_pos is None:
            selected.append(record)
            last_pos, last_quat = pos.copy(), quat.copy()
            continue

        dt = float(np.linalg.norm(pos - last_pos))
        dr = float(_quat_angle_rad(quat, last_quat))

        if dt >= min_translation or dr >= np.deg2rad(min_rotation_deg):
            selected.append(record)
            last_pos, last_quat = pos.copy(), quat.copy()

    return selected


def _quat_angle_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance (radians) between two quaternions [x,y,z,w]."""
    dot = float(abs(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return 2.0 * np.arccos(min(1.0, dot))
