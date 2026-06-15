"""Data I/O for LiDAR-SLAM reconstruction."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

from reconstruction.types import ScanFrame

logger = logging.getLogger(__name__)

_MIN_RANGE = 0.5    # metres – discard returns closer than this (body/mount)
_MAX_RANGE = 100.0  # metres – discard returns beyond this


def load_scan_frames(session_path: Path) -> List[ScanFrame]:
    """
    Discover all PLY scans in session/lidar/ and pair each with a timestamp.

    Timestamps come from JSON sidecars (frame_id, timestamp).
    If sidecars are absent the frame index divided by 10 Hz is used as fallback.
    """
    lidar_path = session_path / "lidar"
    if not lidar_path.exists():
        raise FileNotFoundError(f"No lidar/ folder in {session_path}")

    ply_files = sorted(lidar_path.glob("*.ply"), key=lambda p: int(p.stem))
    if not ply_files:
        raise FileNotFoundError(f"No PLY scans found in {lidar_path}")

    ts_map: dict[int, float] = {}
    for jf in lidar_path.glob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
            ts_map[int(data["frame_id"])] = float(data["timestamp"])
        except Exception:
            pass

    frames: List[ScanFrame] = []
    for ply in ply_files:
        fid = int(ply.stem)
        ts = ts_map.get(fid, float(fid) / 10.0)
        frames.append(ScanFrame(frame_id=fid, path=ply, timestamp=ts))

    logger.info(f"Found {len(frames)} LiDAR scans")
    return frames


def load_point_cloud(path: Path) -> np.ndarray:
    """Load a PLY file → Nx3 float32 array, range-filtered."""
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        return pts
    ranges = np.linalg.norm(pts, axis=1)
    return pts[(ranges >= _MIN_RANGE) & (ranges <= _MAX_RANGE)]


def load_point_cloud_o3d(path: Path, voxel_size: float = 0.1) -> o3d.geometry.PointCloud:
    """Load a PLY file as an Open3D PointCloud and voxel-downsample it."""
    pts = load_point_cloud(path)
    pcd = o3d.geometry.PointCloud()
    if len(pts) == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd.voxel_down_sample(voxel_size)


def load_spot_poses(session_path: Path) -> Optional[np.ndarray]:
    """
    Load poses.npy → Nx8 [ts, x, y, z, qx, qy, qz, qw].
    Returns None if the file is absent.
    """
    p = session_path / "poses.npy"
    if not p.exists():
        logger.warning("poses.npy not found – running without VIO warm-start")
        return None
    arr = np.load(p)
    logger.info(f"Loaded {len(arr)} SPOT VIO poses")
    return arr


def pose_row_to_matrix(row: np.ndarray) -> np.ndarray:
    """[ts, x, y, z, qx, qy, qz, qw] → 4x4 SE(3)."""
    T = np.eye(4)
    T[:3, 3] = row[1:4]
    T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()   # scipy: scalar-last [x,y,z,w]
    return T


def interpolate_spot_pose(timestamp: float, spot_poses: np.ndarray) -> np.ndarray:
    """Return the interpolated 4x4 world pose at the given Unix timestamp."""
    ts = spot_poses[:, 0]
    idx = int(np.searchsorted(ts, timestamp))
    idx = max(1, min(idx, len(spot_poses) - 1))

    t0, t1 = ts[idx - 1], ts[idx]
    alpha = float((timestamp - t0) / (t1 - t0)) if t1 != t0 else 0.0
    alpha = max(0.0, min(1.0, alpha))

    p0, p1 = spot_poses[idx - 1, 1:4], spot_poses[idx, 1:4]
    q0, q1 = spot_poses[idx - 1, 4:8], spot_poses[idx, 4:8]

    pos = (1.0 - alpha) * p0 + alpha * p1
    slerp = Slerp([0.0, 1.0], Rotation.from_quat(np.stack([q0, q1])))
    rot = slerp(float(alpha)).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T
