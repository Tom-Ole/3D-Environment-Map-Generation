"""SE(3) transform utilities and frame-tree helpers."""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (scalar-last convention: [x, y, z, w]).
    Result: q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([x, y, z, w])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Invert a quaternion (scalar-last convention)."""
    x, y, z, w = q
    norm_sq = x * x + y * y + z * z + w * w
    return np.array([-x, -y, -z, w]) / norm_sq


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (scalar-last: [x, y, z, w]) to 3x3 rotation matrix."""
    # scipy uses scalar-first convention, so we need to convert
    q_scipy = np.array([q[3], q[0], q[1], q[2]])  # [w, x, y, z]
    rot = Rotation.from_quat(q_scipy)
    return rot.as_matrix()


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (scalar-last: [x, y, z, w])."""
    rot = Rotation.from_matrix(R)
    q_scipy = rot.as_quat()  # [x, y, z, w] from scipy
    return q_scipy


def compose_transforms(
    pos1: np.ndarray, quat1: np.ndarray, pos2: np.ndarray, quat2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose two SE(3) transforms: T1 * T2.

    Args:
        pos1, quat1: First transform (position in 3D, quaternion scalar-last)
        pos2, quat2: Second transform

    Returns:
        Composed position and quaternion
    """
    R1 = quaternion_to_rotation_matrix(quat1)
    pos_composed = pos1 + R1 @ pos2
    quat_composed = quaternion_multiply(quat1, quat2)
    return pos_composed, quat_composed


def invert_transform(
    pos: np.ndarray, quat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert an SE(3) transform.

    Args:
        pos: Position in 3D
        quat: Quaternion (scalar-last)

    Returns:
        Inverted position and quaternion
    """
    quat_inv = quaternion_inverse(quat)
    R_inv = quaternion_to_rotation_matrix(quat_inv)
    pos_inv = R_inv @ (-pos)
    return pos_inv, quat_inv


def interpolate_pose(
    pos1: np.ndarray,
    quat1: np.ndarray,
    pos2: np.ndarray,
    quat2: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear interpolation of positions, SLERP of rotations.

    Args:
        pos1, quat1: Start pose
        pos2, quat2: End pose
        alpha: Interpolation parameter [0, 1]

    Returns:
        Interpolated position and quaternion
    """
    pos_interp = (1 - alpha) * pos1 + alpha * pos2

    # SLERP for rotation (scipy uses scalar-first)
    q1_scipy = np.array([quat1[3], quat1[0], quat1[1], quat1[2]])
    q2_scipy = np.array([quat2[3], quat2[0], quat2[1], quat2[2]])

    slerp = Slerp([0, 1], Rotation.from_quat(np.array([q1_scipy, q2_scipy])))
    quat_interp_scipy = slerp(alpha).as_quat()

    # Convert back to scalar-last
    quat_interp = np.array(
        [
            quat_interp_scipy[1],
            quat_interp_scipy[2],
            quat_interp_scipy[3],
            quat_interp_scipy[0],
        ]
    )

    return pos_interp, quat_interp


def align_odometry_to_reference(
    odometry_poses: np.ndarray, reference_poses: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Align an odometry trajectory to reference poses using Umeyama's method.

    Args:
        odometry_poses: Nx7 array [t, x, y, z, qx, qy, qz, qw]
        reference_poses: Nx7 array (reference, typically sparser)

    Returns:
        Aligned odometry poses (Nx7) and alignment error
    """
    odo_pos = odometry_poses[:, 1:4]
    ref_pos = reference_poses[:, 1:4]

    # Subsample the larger trajectory to match the smaller one before SVD.
    # SPOT provides sparser poses (e.g. 81) vs dense odometry (e.g. 481).
    n_odo, n_ref = len(odo_pos), len(ref_pos)
    if n_odo != n_ref:
        if n_odo > n_ref:
            idx = np.round(np.linspace(0, n_odo - 1, n_ref)).astype(int)
            odo_pos_aligned = odo_pos[idx]
            ref_pos_aligned = ref_pos
        else:
            idx = np.round(np.linspace(0, n_ref - 1, n_odo)).astype(int)
            odo_pos_aligned = odo_pos
            ref_pos_aligned = ref_pos[idx]
    else:
        odo_pos_aligned = odo_pos
        ref_pos_aligned = ref_pos

    # Simple rigid alignment using SVD (Umeyama)
    centroid_odo = odo_pos_aligned.mean(axis=0)
    centroid_ref = ref_pos_aligned.mean(axis=0)

    H = (odo_pos_aligned - centroid_odo).T @ (ref_pos_aligned - centroid_ref)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_ref - R @ centroid_odo

    R_quat = rotation_matrix_to_quaternion(R)
    aligned_poses = odometry_poses.copy()
    for i in range(len(aligned_poses)):
        aligned_poses[i, 1:4] = R @ odometry_poses[i, 1:4] + t
        aligned_poses[i, 4:8] = quaternion_multiply(R_quat, odometry_poses[i, 4:8])

    error = np.mean(np.linalg.norm(R @ odo_pos_aligned.T + t[:, None] - ref_pos_aligned.T, axis=0))
    return aligned_poses, error
