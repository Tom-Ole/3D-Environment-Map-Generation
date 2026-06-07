"""Multi-stream timestamp synchronization utilities."""

from typing import List, Optional, Tuple

import numpy as np


def synchronize_timestamps(
    target_timestamp: float,
    source_timestamps: np.ndarray,
    data: np.ndarray,
    method: str = "nearest",
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Find and return data corresponding to a target timestamp.

    Args:
        target_timestamp: Target Unix timestamp
        source_timestamps: Array of timestamps for the data
        data: Data array (first dimension must match timestamps)
        method: "nearest" for nearest neighbor, "linear" for interpolation

    Returns:
        Synchronized data (or None if not found) and index
    """
    if len(source_timestamps) == 0:
        return None, None

    # Find nearest timestamp
    idx = np.argmin(np.abs(source_timestamps - target_timestamp))
    dt = abs(source_timestamps[idx] - target_timestamp)

    if method == "nearest":
        return data[idx], idx

    elif method == "linear":
        # Find bracketing timestamps
        if source_timestamps[idx] < target_timestamp:
            if idx + 1 < len(source_timestamps):
                idx_next = idx + 1
            else:
                return data[idx], idx
        else:
            if idx > 0:
                idx_prev = idx
                idx = idx - 1
                idx_next = idx_prev
            else:
                return data[idx], idx

        t1, t2 = source_timestamps[idx], source_timestamps[idx_next]
        if t1 == t2:
            return data[idx], idx

        alpha = (target_timestamp - t1) / (t2 - t1)
        alpha = max(0.0, min(1.0, alpha))

        # Linear interpolation
        interp_data = (1 - alpha) * data[idx] + alpha * data[idx_next]
        return interp_data, idx

    return None, None


def align_streams(
    stream_timestamps: dict,  # {"stream_name": np.ndarray of timestamps}
    reference_stream: str,
) -> dict:
    """
    Align all streams to a reference stream by finding common timestamps.

    Args:
        stream_timestamps: Dict mapping stream names to timestamp arrays
        reference_stream: Name of reference stream

    Returns:
        Dict mapping stream names to lists of aligned indices
    """
    if reference_stream not in stream_timestamps:
        raise ValueError(f"Reference stream '{reference_stream}' not found")

    ref_times = stream_timestamps[reference_stream]
    aligned_indices = {reference_stream: np.arange(len(ref_times))}

    for stream_name, times in stream_timestamps.items():
        if stream_name == reference_stream:
            continue

        indices = []
        for ref_t in ref_times:
            if len(times) == 0:
                indices.append(None)
            else:
                idx = np.argmin(np.abs(times - ref_t))
                if np.abs(times[idx] - ref_t) < 1.0:  # Within 1 second
                    indices.append(idx)
                else:
                    indices.append(None)

        aligned_indices[stream_name] = indices

    return aligned_indices


def interpolate_pose_to_timestamp(
    target_timestamp: float,
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Interpolate a pose to a specific timestamp.

    Args:
        target_timestamp: Desired timestamp
        timestamps: Array of pose timestamps
        positions: Nx3 array of positions
        quaternions: Nx4 array of quaternions (scalar-last)

    Returns:
        Interpolated position and quaternion, or (None, None) if out of range
    """
    if len(timestamps) == 0:
        return None, None

    if target_timestamp < timestamps[0] or target_timestamp > timestamps[-1]:
        return None, None

    idx = np.searchsorted(timestamps, target_timestamp)

    if idx == 0:
        return positions[0].copy(), quaternions[0].copy()
    elif idx == len(timestamps):
        return positions[-1].copy(), quaternions[-1].copy()

    idx_prev = idx - 1
    idx_next = idx

    t1, t2 = timestamps[idx_prev], timestamps[idx_next]
    if t1 == t2:
        return positions[idx_prev].copy(), quaternions[idx_prev].copy()

    alpha = (target_timestamp - t1) / (t2 - t1)

    # Linear interpolation for position
    pos_interp = (1 - alpha) * positions[idx_prev] + alpha * positions[idx_next]

    # SLERP for quaternion
    from scipy.spatial.transform import Rotation, Slerp

    q1_scipy = np.array([quaternions[idx_prev, 3], quaternions[idx_prev, 0], quaternions[idx_prev, 1], quaternions[idx_prev, 2]])
    q2_scipy = np.array([quaternions[idx_next, 3], quaternions[idx_next, 0], quaternions[idx_next, 1], quaternions[idx_next, 2]])

    slerp = Slerp([0, 1], Rotation.from_quat(np.array([q1_scipy, q2_scipy])))
    quat_interp_scipy = slerp(alpha).as_quat()

    # Convert back to scalar-last
    quat_interp = np.array(
        [quat_interp_scipy[1], quat_interp_scipy[2], quat_interp_scipy[3], quat_interp_scipy[0]]
    )

    return pos_interp, quat_interp


def find_time_gap_indices(timestamps: np.ndarray, max_gap: float) -> List[int]:
    """
    Find indices where there are large time gaps (e.g., recording pauses).

    Args:
        timestamps: Array of timestamps
        max_gap: Maximum acceptable gap in seconds

    Returns:
        List of indices where gaps occur
    """
    if len(timestamps) < 2:
        return []

    gaps = np.diff(timestamps)
    gap_indices = np.where(gaps > max_gap)[0] + 1
    return gap_indices.tolist()
