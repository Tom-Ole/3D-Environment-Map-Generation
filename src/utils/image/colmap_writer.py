"""
colmap_writer.py  —  Fixed & extended version
==============================================
Writes a COLMAP text-format sparse model (cameras.txt / images.txt / points3D.txt)
from externally supplied camera intrinsics and cam-to-world poses.

Fixes over the original
-----------------------
1. Camera model: supports OPENCV_FISHEYE (8 params) and FULL_OPENCV (12 params)
   in addition to PINHOLE (4 params).  OPENCV_FISHEYE is the correct model for
   Boston Dynamics Spot fisheye cameras.

2. Pose inversion: replaced np.linalg.inv() with the exact closed-form rigid-body
   inverse  (R_wc = R_c2w.T,  t_wc = -R_wc @ t_c2w).  The LU-based general
   inverse accumulates O(1e-14) per-element error; the closed form is exact.

3. Frame-convention rotation: matrix_to_colmap_pose() accepts an optional
   frame_R (3×3 ndarray).  Pass SPOT_FLU_TO_COLMAP_RDF to convert Spot's
   vision-frame poses (X-forward, Y-left, Z-up) into COLMAP's world frame
   before computing the world-to-camera extrinsic.

4. add_image() / write_image() forward frame_R to matrix_to_colmap_pose().

5. validate_colmap_txt_model() accepts PINHOLE (4), OPENCV_FISHEYE (8), and
   FULL_OPENCV (12) parameter counts; no longer hard-asserts PINHOLE only.

6. Blank-line structure in images.txt: the two-line-per-image spec is enforced
   (pose line + empty POINTS2D line, nothing else between blocks).

Format reference
----------------
https://colmap.github.io/legacy/3.9/format.html
https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h

Key conventions (verified against source / docs)
-------------------------------------------------
cameras.txt   : CAMERA_ID  MODEL  WIDTH  HEIGHT  PARAMS[]
                PINHOLE params          : fx fy cx cy
                OPENCV_FISHEYE params   : fx fy cx cy k1 k2 k3 k4
                FULL_OPENCV params      : fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6

images.txt    : TWO lines per image (no extra blank lines between blocks)
                line 1 — IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                line 2 — POINTS2D[] as (X Y POINT3D_ID); intentionally empty

Quaternion    : Hamilton convention, world-to-camera, scalar-first (QW QX QY QZ)
                scipy.as_quat() returns (x,y,z,w); we unpack as qx,qy,qz,qw and
                write qw first.

IDs           : 1-based (CAMERA_ID and IMAGE_ID both start at 1).
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame-convention constants
# ---------------------------------------------------------------------------

# Rotation that maps Spot's FLU world frame (X-forward, Y-left, Z-up)
# to COLMAP's expected world frame so that camera optical axes are expressed
# in a consistent right-handed coordinate system.
#
# Derivation:
#   COLMAP camera X (right)    = -Spot camera Y → row 0 = [ 0,-1, 0]
#   COLMAP camera Y (down)     = -Spot camera Z → row 1 = [ 0, 0,-1]
#   COLMAP camera Z (forward)  = +Spot camera X → row 2 = [ 1, 0, 0]
#
# Usage:
#   writer.add_image(name, cam_id, T_world_cam_spot,
#                    frame_R=SPOT_FLU_TO_COLMAP_RDF)
SPOT_FLU_TO_COLMAP_RDF: np.ndarray = np.array(
    [
        [ 0.0, -1.0,  0.0],   # COLMAP X (right)    = -Spot Y (left in FLU → right is -Y)
        [ 0.0,  0.0, -1.0],   # COLMAP Y (down)     = -Spot Z (up in FLU  → down  is -Z)
        [ 1.0,  0.0,  0.0],   # COLMAP Z (forward)  = +Spot X (forward in FLU)
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Camera model registry
# ---------------------------------------------------------------------------

# Maps COLMAP model name → ordered list of parameter keys.
# The order matches the PARAMS[] column order in cameras.txt.
CAMERA_MODEL_PARAMS: Dict[str, List[str]] = {
    "PINHOLE": [
        "fx", "fy", "cx", "cy",
    ],
    "OPENCV_FISHEYE": [
        "fx", "fy", "cx", "cy",
        "k1", "k2", "k3", "k4",
    ],
    "FULL_OPENCV": [
        "fx", "fy", "cx", "cy",
        "k1", "k2", "p1", "p2",
        "k3", "k4", "k5", "k6",
    ],
}

# Keys that must be explicitly supplied (not allowed to default to 0).
_REQUIRED_KEYS: Set[str] = {"fx", "fy"}

# Keys whose default value is image-dimension-dependent (not 0).
_PRINCIPAL_POINT_KEYS: Set[str] = {"cx", "cy"}

# All remaining keys (distortion) default to 0.0 when absent.


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_rotation_matrix(R: np.ndarray, *, label: str = "R") -> None:
    """
    Raise ValueError if *R* is not a valid 3×3 proper rotation matrix.

    Checks: shape, finiteness, orthonormality (R^T R ≈ I, tol=1e-5),
    and determinant ≈ +1.
    """
    if R.shape != (3, 3):
        raise ValueError(f"{label}: expected shape (3,3), got {R.shape}")
    if not np.isfinite(R).all():
        raise ValueError(f"{label}: contains non-finite values (NaN or Inf)")
    eye_diff = np.max(np.abs(R.T @ R - np.eye(3)))
    if eye_diff > 1e-5:
        raise ValueError(
            f"{label}: columns are not orthonormal "
            f"— max deviation from I: {eye_diff:.2e}"
        )
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-5:
        raise ValueError(
            f"{label}: determinant is {det:.6f} (expected +1.0 for a proper rotation)"
        )


def _validate_cam_to_world(cam_to_world: np.ndarray) -> None:
    """
    Raise ValueError if *cam_to_world* is not a valid 4×4 rigid-body transform.
    """
    if cam_to_world.shape != (4, 4):
        raise ValueError(
            f"cam_to_world: expected shape (4,4), got {cam_to_world.shape}"
        )
    if not np.isfinite(cam_to_world).all():
        raise ValueError("cam_to_world: contains non-finite values (NaN or Inf)")
    bottom = cam_to_world[3, :]
    if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(
            f"cam_to_world: bottom row must be [0,0,0,1], got {bottom}"
        )
    _validate_rotation_matrix(cam_to_world[:3, :3], label="cam_to_world[:3,:3]")


def _validate_frame_R(frame_R: np.ndarray) -> None:
    """
    Raise ValueError if *frame_R* is not a valid 3×3 rotation matrix.
    """
    _validate_rotation_matrix(frame_R, label="frame_R")


# ---------------------------------------------------------------------------
# Core pose conversion
# ---------------------------------------------------------------------------

def matrix_to_colmap_pose(
    cam_to_world: np.ndarray,
    *,
    frame_R: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Convert a 4×4 camera-to-world matrix to COLMAP's extrinsic representation.

    COLMAP stores the *world-to-camera* transform:

        X_cam = R_wc @ X_world + t_wc

    as a 7-tuple (QW, QX, QY, QZ, TX, TY, TZ) using the Hamilton quaternion
    convention with the scalar part (w) written first — matching the images.txt
    column order.

    Parameters
    ----------
    cam_to_world : ndarray, shape (4, 4)
        Camera-to-world rigid transform.
        The camera centre in world coordinates is ``cam_to_world[:3, 3]``.

    frame_R : ndarray, shape (3, 3), optional
        An additional rotation applied to the *world* coordinate frame before
        computing the world-to-camera transform.  Use this to account for
        axis-convention differences between the source frame (e.g. Spot's
        vision frame, which is X-forward / Y-left / Z-up) and COLMAP's
        expected frame orientation.

        The transform applied is::

            R_c2w_colmap = frame_R @ R_c2w_src @ frame_R.T
            t_c2w_colmap = frame_R @ t_c2w_src

        Then the world-to-camera inverse is computed from these rotated values.

        Supply ``SPOT_FLU_TO_COLMAP_RDF`` for Spot datasets.

    Returns
    -------
    (qw, qx, qy, qz, tx, ty, tz) : tuple of float
        COLMAP extrinsic parameters (world-to-camera, Hamilton quaternion,
        scalar first).

    Raises
    ------
    ValueError
        If ``cam_to_world`` is not a valid 4×4 rigid-body transform, or
        if ``frame_R`` is not a valid 3×3 rotation matrix.

    Notes
    -----
    The world-to-camera inversion uses the **exact closed-form** rigid-body
    inverse:

        R_wc = R_c2w.T
        t_wc = -R_wc @ t_c2w

    This is algebraically exact (no LU decomposition, no floating-point
    accumulation), unlike ``np.linalg.inv()`` which accumulates O(1e-14)
    per-element error and can trigger the orthonormality check for long
    trajectories.

    scipy's ``Rotation.as_quat()`` returns (x, y, z, w).  We unpack into
    named variables (qx, qy, qz, qw) and then reorder to produce
    (qw, qx, qy, qz) as COLMAP's images.txt requires.

    The camera centre in world coordinates is always:

        C_world = -R_wc.T @ t_wc = cam_to_world[:3, 3]
    """
    _validate_cam_to_world(cam_to_world)

    R_c2w: np.ndarray = cam_to_world[:3, :3]
    t_c2w: np.ndarray = cam_to_world[:3, 3]

    # Apply optional world-frame rotation (axis-convention change).
    if frame_R is not None:
        _validate_frame_R(frame_R)
        R_c2w = frame_R @ R_c2w @ frame_R.T
        t_c2w = frame_R @ t_c2w

    # Exact closed-form rigid-body inverse.
    R_wc: np.ndarray = R_c2w.T
    t_wc: np.ndarray = -R_wc @ t_c2w

    # scipy returns (x, y, z, w); unpack into named variables then reorder.
    qx, qy, qz, qw = Rotation.from_matrix(R_wc).as_quat()

    return (
        float(qw), float(qx), float(qy), float(qz),
        float(t_wc[0]), float(t_wc[1]), float(t_wc[2]),
    )


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class _CameraRecord:
    """Internal storage for one registered camera."""

    __slots__ = (
        "camera_id", "source_name", "model",
        "width", "height", "params",
    )

    def __init__(
        self,
        camera_id: int,
        source_name: str,
        model: str,
        width: int,
        height: int,
        params: List[float],
    ) -> None:
        self.camera_id = camera_id
        self.source_name = source_name
        self.model = model
        self.width = width
        self.height = height
        self.params = params  # ordered list matching CAMERA_MODEL_PARAMS[model]


class _ImageRecord:
    """Internal storage for one buffered image."""

    __slots__ = (
        "image_id", "image_name", "camera_id",
        "qw", "qx", "qy", "qz",
        "tx", "ty", "tz",
    )

    def __init__(
        self,
        image_id: int,
        image_name: str,
        camera_id: int,
        qw: float, qx: float, qy: float, qz: float,
        tx: float, ty: float, tz: float,
    ) -> None:
        self.image_id = image_id
        self.image_name = image_name
        self.camera_id = camera_id
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.tx = tx
        self.ty = ty
        self.tz = tz


# ---------------------------------------------------------------------------
# Main writer class
# ---------------------------------------------------------------------------

class ColmapWriter:
    """
    Buffer camera intrinsics and image poses, then write a valid COLMAP
    text-format sparse model on :meth:`save`.

    The three text files required for COLMAP import are produced:

    * ``cameras.txt``   — intrinsics (PINHOLE, OPENCV_FISHEYE, or FULL_OPENCV)
    * ``images.txt``    — world-to-camera extrinsics (no 2D keypoints)
    * ``points3D.txt``  — empty; COLMAP fills this during reconstruction

    Quick start — Spot fisheye pipeline
    ------------------------------------
    ::

        from colmap_writer import ColmapWriter, SPOT_FLU_TO_COLMAP_RDF

        writer = ColmapWriter(sparse_dir)

        # Register each physical camera once, using BD SDK calibration values.
        cam_back = writer.register_camera(
            "back_fisheye_image",
            width=640, height=480,
            model="OPENCV_FISHEYE",
            intrinsics={
                "fx": 330.73, "fy": 331.12, "cx": 311.38, "cy": 241.71,
                "k1": -0.013,  "k2": 0.0007, "k3": -3e-5,  "k4": 0.0,
            },
        )

        # For every captured frame, supply the cam-to-world transform in
        # Spot's vision frame and pass frame_R to rotate into COLMAP's frame.
        writer.add_image(
            "back_fisheye_image/00000.jpg",
            cam_back,
            cam_to_world=T_world_camera_spot,
            frame_R=SPOT_FLU_TO_COLMAP_RDF,
        )

        writer.save()

    Context-manager form::

        with ColmapWriter(sparse_dir) as w:
            cam_id = w.register_camera(...)
            w.add_image(..., frame_R=SPOT_FLU_TO_COLMAP_RDF)

    Parameters
    ----------
    sparse_dir : path-like
        Destination directory.  Created (including parents) if absent.
    default_model : str, optional
        Camera model used when ``register_camera()`` is called without an
        explicit *model* argument.  Defaults to ``"OPENCV_FISHEYE"``.
    default_frame_R : ndarray (3, 3), optional
        Frame rotation applied in every ``add_image()`` call when no
        per-call *frame_R* is supplied.  Set once at construction time
        to avoid repeating the argument on every call.  Pass
        ``SPOT_FLU_TO_COLMAP_RDF`` for Spot datasets.
    """

    def __init__(
        self,
        sparse_dir,
        *,
        default_model: str = "OPENCV_FISHEYE",
        default_frame_R: Optional[np.ndarray] = None,
    ) -> None:
        self._sparse_dir = Path(sparse_dir)
        self._sparse_dir.mkdir(parents=True, exist_ok=True)

        if default_model not in CAMERA_MODEL_PARAMS:
            raise ValueError(
                f"default_model '{default_model}' is not supported. "
                f"Choose from: {list(CAMERA_MODEL_PARAMS)}"
            )
        self._default_model = default_model

        if default_frame_R is not None:
            _validate_frame_R(default_frame_R)
        self._default_frame_R: Optional[np.ndarray] = default_frame_R

        # source_name → _CameraRecord
        self._cameras: Dict[str, _CameraRecord] = {}
        self._next_camera_id: int = 1

        # Insertion-ordered list of _ImageRecord
        self._images: List[_ImageRecord] = []
        self._next_image_id: int = 1

        # Guards
        self._saved: bool = False
        self._image_names_seen: Set[str] = set()

    # --------------------------------------------------------------- cameras

    def register_camera(
        self,
        source_name: str,
        width: int,
        height: int,
        intrinsics: dict,
        model: Optional[str] = None,
    ) -> int:
        """
        Register a camera and return its COLMAP ``CAMERA_ID``.

        Calling this method again with the same *source_name* is idempotent —
        the existing ID is returned without modifying any state.

        Parameters
        ----------
        source_name : str
            Arbitrary label identifying this camera's configuration (e.g.
            ``"back_fisheye_image"``).  Multiple images may share the same
            camera by passing the returned ID to :meth:`add_image`.
        width, height : int
            Sensor dimensions in pixels.  Must be positive.
        intrinsics : dict
            Camera intrinsic parameters.  Required keys depend on *model*:

            PINHOLE (4 params)
                ``"fx"``, ``"fy"`` required; ``"cx"`` / ``"cy"`` optional
                (default: image centre).
            OPENCV_FISHEYE (8 params)
                ``"fx"``, ``"fy"`` required; ``"cx"`` / ``"cy"`` optional;
                ``"k1"``, ``"k2"``, ``"k3"``, ``"k4"`` optional (default 0).
                Get k1–k4 from BD SDK ``SpotCameraCalibration`` proto.
            FULL_OPENCV (12 params)
                ``"fx"``, ``"fy"`` required; ``"cx"`` / ``"cy"`` optional;
                ``"k1"``–``"k6"``, ``"p1"``, ``"p2"`` optional (default 0).

        model : str, optional
            COLMAP camera model string.  Defaults to the *default_model*
            supplied at construction (``"OPENCV_FISHEYE"`` unless overridden).

        Returns
        -------
        int
            Assigned COLMAP ``CAMERA_ID`` (1-based).

        Raises
        ------
        TypeError
            If *source_name* is not a str or dimensions are not integers.
        ValueError
            If dimensions are non-positive, *model* is unsupported,
            ``fx``/``fy`` are missing or non-positive.
        """
        if not isinstance(source_name, str):
            raise TypeError(
                f"source_name must be str, got {type(source_name).__name__}"
            )
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("width and height must be integers")
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive, got {width}×{height}"
            )

        # Idempotent registration.
        if source_name in self._cameras:
            existing = self._cameras[source_name]
            logger.debug(
                "register_camera: '%s' already registered as camera_id=%d",
                source_name, existing.camera_id,
            )
            return existing.camera_id

        # Resolve model.
        resolved_model = model if model is not None else self._default_model
        if resolved_model not in CAMERA_MODEL_PARAMS:
            raise ValueError(
                f"Unsupported camera model '{resolved_model}'. "
                f"Supported: {list(CAMERA_MODEL_PARAMS)}"
            )
        param_keys = CAMERA_MODEL_PARAMS[resolved_model]

        # Build ordered parameter list.
        params: List[float] = []
        for key in param_keys:
            if key in _REQUIRED_KEYS:
                val = intrinsics.get(key)
                if val is None:
                    raise ValueError(
                        f"intrinsics for '{source_name}' must contain '{key}'"
                    )
                fval = float(val)
                if fval <= 0.0:
                    raise ValueError(
                        f"'{key}' must be positive for '{source_name}'; got {fval}"
                    )
                params.append(fval)
            elif key == "cx":
                params.append(float(intrinsics.get("cx", width / 2.0)))
            elif key == "cy":
                params.append(float(intrinsics.get("cy", height / 2.0)))
            else:
                # Distortion coefficients default to 0.
                params.append(float(intrinsics.get(key, 0.0)))

        camera_id = self._next_camera_id
        self._next_camera_id += 1
        self._cameras[source_name] = _CameraRecord(
            camera_id=camera_id,
            source_name=source_name,
            model=resolved_model,
            width=width,
            height=height,
            params=params,
        )
        logger.debug(
            "Registered camera %d: '%s' %s %dx%d  params=%s",
            camera_id, source_name, resolved_model, width, height,
            [f"{p:.6g}" for p in params],
        )
        return camera_id

    # ----------------------------------------------------------------- images

    def add_image(
        self,
        image_name: str,
        camera_id: int,
        cam_to_world: np.ndarray,
        *,
        frame_R: Optional[np.ndarray] = None,
    ) -> int:
        """
        Buffer one image entry for writing.

        Parameters
        ----------
        image_name : str
            Path relative to the COLMAP images folder, e.g.
            ``"back_fisheye_image/00042.jpg"``.  Must be unique.
        camera_id : int
            COLMAP ``CAMERA_ID`` returned by :meth:`register_camera`.
        cam_to_world : ndarray, shape (4, 4)
            Camera-to-world rigid transform (in the source coordinate frame).
            ``cam_to_world[:3, 3]`` is the camera centre in world coordinates.
        frame_R : ndarray (3, 3), optional
            Per-call frame rotation.  If supplied, overrides the instance-level
            *default_frame_R*.  If neither is set, no frame rotation is applied.
            Pass ``SPOT_FLU_TO_COLMAP_RDF`` for Spot vision-frame poses.

        Returns
        -------
        int
            Assigned COLMAP ``IMAGE_ID`` (1-based).

        Raises
        ------
        ValueError
            If *camera_id* is not registered, *image_name* is a duplicate,
            or *cam_to_world* / *frame_R* fail validation.
        """
        # Validate camera_id.
        registered_ids = {rec.camera_id for rec in self._cameras.values()}
        if camera_id not in registered_ids:
            raise ValueError(
                f"camera_id={camera_id} has not been registered. "
                f"Registered IDs: {sorted(registered_ids)}"
            )

        # Validate image_name uniqueness.
        if image_name in self._image_names_seen:
            raise ValueError(
                f"Duplicate image_name: '{image_name}' has already been added"
            )

        # Resolve frame rotation: per-call > instance default > None.
        resolved_frame_R = frame_R if frame_R is not None else self._default_frame_R

        # Convert pose (also validates cam_to_world and frame_R internally).
        qw, qx, qy, qz, tx, ty, tz = matrix_to_colmap_pose(
            cam_to_world, frame_R=resolved_frame_R
        )

        image_id = self._next_image_id
        self._next_image_id += 1
        self._image_names_seen.add(image_name)
        self._images.append(
            _ImageRecord(
                image_id=image_id,
                image_name=image_name,
                camera_id=camera_id,
                qw=qw, qx=qx, qy=qy, qz=qz,
                tx=tx, ty=ty, tz=tz,
            )
        )
        logger.debug(
            "Buffered image %d: '%s' (camera_id=%d)", image_id, image_name, camera_id
        )
        return image_id

    def write_image(
        self,
        image_name: str,
        camera_id: int,
        cam_to_world: np.ndarray,
        *,
        frame_R: Optional[np.ndarray] = None,
    ) -> int:
        """Alias for :meth:`add_image` (backward compatibility)."""
        return self.add_image(image_name, camera_id, cam_to_world, frame_R=frame_R)

    # ------------------------------------------------------------------ save

    def save(self) -> None:
        """
        Write ``cameras.txt``, ``images.txt``, and ``points3D.txt`` to the
        directory supplied at construction time.

        Raises
        ------
        RuntimeError
            If called more than once on the same instance.
        """
        if self._saved:
            raise RuntimeError(
                "ColmapWriter.save() has already been called on this instance. "
                "Create a new ColmapWriter (or call reset()) to write another model."
            )
        self._saved = True

        self._write_cameras_txt()
        self._write_images_txt()
        self._write_points3d_txt()

        logger.info(
            "COLMAP model saved → %s  (%d camera(s), %d image(s))",
            self._sparse_dir, len(self._cameras), len(self._images),
        )

    def reset(self) -> None:
        """
        Clear all buffered cameras and images so the writer can be reused
        for a new session without constructing a new instance.

        The destination directory is preserved; the next :meth:`save` will
        overwrite any files previously written there.
        """
        self._cameras.clear()
        self._next_camera_id = 1
        self._images.clear()
        self._next_image_id = 1
        self._image_names_seen.clear()
        self._saved = False
        logger.debug("ColmapWriter reset — ready for new session (%s)", self._sparse_dir)

    # -------------------------------------------------------- context manager

    def __enter__(self) -> "ColmapWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.save()
        return False

    # ---------------------------------------------------------------- queries

    @property
    def num_cameras(self) -> int:
        """Number of registered cameras."""
        return len(self._cameras)

    @property
    def num_images(self) -> int:
        """Number of buffered images."""
        return len(self._images)

    # ------------------------------------------------------- txt file writers

    def _write_cameras_txt(self) -> None:
        """
        Write ``cameras.txt``.

        Format (one line per camera)::

            # Camera list with one line of data per camera:
            #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            # Number of cameras: N
            CAMERA_ID  MODEL  WIDTH  HEIGHT  param0 param1 ...

        PARAMS[] order per model (from COLMAP models.h):
            PINHOLE         : fx fy cx cy
            OPENCV_FISHEYE  : fx fy cx cy k1 k2 k3 k4
            FULL_OPENCV     : fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
        """
        header = (
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"# Number of cameras: {len(self._cameras)}\n"
        )
        lines = [header]
        for rec in self._cameras.values():
            params_str = " ".join(f"{p:.10g}" for p in rec.params)
            lines.append(
                f"{rec.camera_id} {rec.model} {rec.width} {rec.height} {params_str}\n"
            )
        (self._sparse_dir / "cameras.txt").write_text("".join(lines), encoding="utf-8")

    def _write_images_txt(self) -> None:
        """
        Write ``images.txt``.

        Format — exactly TWO lines per image, no extra blank lines between
        blocks (COLMAP's C++ reader is tolerant of extras, but pycolmap,
        nerfstudio, and colmap2nerf are not)::

            # Image list with two lines of data per image:
            #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            #   POINTS2D[] as (X, Y, POINT3D_ID)
            # Number of images: N, mean observations per image: 0
            IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            <empty POINTS2D line>
            IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            <empty POINTS2D line>
            ...

        Quaternion convention: Hamilton, world-to-camera, scalar (w) first.
        The POINTS2D line is intentionally empty for prior-pose import.
        """
        header = (
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {len(self._images)}, mean observations per image: 0\n"
        )
        lines = [header]
        for rec in self._images:
            # Line 1: pose + metadata.
            lines.append(
                f"{rec.image_id} "
                f"{rec.qw:.10g} {rec.qx:.10g} {rec.qy:.10g} {rec.qz:.10g} "
                f"{rec.tx:.10g} {rec.ty:.10g} {rec.tz:.10g} "
                f"{rec.camera_id} {rec.image_name}\n"
            )
            # Line 2: empty POINTS2D list (required by the two-line-per-image spec).
            lines.append("\n")
        (self._sparse_dir / "images.txt").write_text("".join(lines), encoding="utf-8")

    def _write_points3d_txt(self) -> None:
        """
        Write an empty ``points3D.txt``.

        COLMAP populates this during feature matching and triangulation.
        The file must exist with valid headers even when empty.
        """
        content = (
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0, mean track length: 0\n"
        )
        (self._sparse_dir / "points3D.txt").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Post-write round-trip validator
# ---------------------------------------------------------------------------

# Maps model name → expected number of PARAMS[] values.
_MODEL_PARAM_COUNTS: Dict[str, int] = {
    name: len(keys) for name, keys in CAMERA_MODEL_PARAMS.items()
}


def validate_colmap_txt_model(sparse_dir) -> None:
    """
    Parse the three COLMAP text files and assert internal consistency.

    Accepts PINHOLE (4 params), OPENCV_FISHEYE (8 params), and FULL_OPENCV
    (12 params).  Intended for use in tests and CI pipelines.

    Checks
    ------
    * All three files exist.
    * Every CAMERA_ID in images.txt exists in cameras.txt.
    * Every IMAGE_ID is unique; every CAMERA_ID is unique.
    * Camera model is a known supported string.
    * Correct number of PARAMS[] for the declared model.
    * fx, fy are positive for every camera.
    * Quaternion norm ≈ 1 for every image (tol = 1e-4).
    * cameras.txt count header matches actual camera count.
    * images.txt count header matches actual image count.
    * points3D.txt exists (data lines are not required to be absent —
      a pre-populated sparse model is valid).

    Parameters
    ----------
    sparse_dir : path-like
        Directory containing ``cameras.txt``, ``images.txt``, ``points3D.txt``.

    Raises
    ------
    AssertionError
        On the first consistency violation found.
    FileNotFoundError
        If any required file is absent.
    """
    sparse_dir = Path(sparse_dir)

    cameras_path  = sparse_dir / "cameras.txt"
    images_path   = sparse_dir / "images.txt"
    points3d_path = sparse_dir / "points3D.txt"

    for p in (cameras_path, images_path, points3d_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    # ---- parse cameras.txt ----
    cameras: Dict[int, dict] = {}
    declared_cam_count: Optional[int] = None

    for line in cameras_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if "Number of cameras:" in stripped:
                try:
                    declared_cam_count = int(stripped.split(":")[-1].strip())
                except ValueError:
                    pass
            continue

        parts = stripped.split()
        assert len(parts) >= 5, f"cameras.txt: malformed line: {line!r}"

        cam_id = int(parts[0])
        assert cam_id not in cameras, f"cameras.txt: duplicate CAMERA_ID {cam_id}"

        model = parts[1]
        assert model in _MODEL_PARAM_COUNTS, (
            f"cameras.txt: unsupported model '{model}' for CAMERA_ID {cam_id}. "
            f"Supported: {list(_MODEL_PARAM_COUNTS)}"
        )

        width, height = int(parts[2]), int(parts[3])
        assert width > 0 and height > 0, (
            f"cameras.txt: non-positive dimensions for CAMERA_ID {cam_id}"
        )

        params = [float(p) for p in parts[4:]]
        expected_count = _MODEL_PARAM_COUNTS[model]
        assert len(params) == expected_count, (
            f"cameras.txt: {model} expects {expected_count} params, "
            f"got {len(params)} for CAMERA_ID {cam_id}"
        )

        # fx=params[0], fy=params[1] for all supported models.
        assert params[0] > 0.0 and params[1] > 0.0, (
            f"cameras.txt: non-positive focal length for CAMERA_ID {cam_id} "
            f"(fx={params[0]}, fy={params[1]})"
        )

        cameras[cam_id] = {
            "model": model, "width": width, "height": height, "params": params
        }

    if declared_cam_count is not None:
        assert declared_cam_count == len(cameras), (
            f"cameras.txt: header says {declared_cam_count} cameras "
            f"but found {len(cameras)}"
        )

    # ---- parse images.txt ----
    image_ids_seen: Set[int] = set()
    declared_img_count: Optional[int] = None
    image_count_actual = 0

    lines = images_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("#"):
            if "Number of images:" in stripped:
                try:
                    declared_img_count = int(
                        stripped.split("Number of images:")[-1]
                        .split(",")[0]
                        .strip()
                    )
                except ValueError:
                    pass
            idx += 1
            continue

        # Line 1 of a two-line image block.
        parts = stripped.split()
        assert len(parts) >= 9, f"images.txt: malformed pose line: {line!r}"

        img_id = int(parts[0])
        assert img_id not in image_ids_seen, (
            f"images.txt: duplicate IMAGE_ID {img_id}"
        )
        image_ids_seen.add(img_id)

        qw = float(parts[1])
        qx = float(parts[2])
        qy = float(parts[3])
        qz = float(parts[4])
        qnorm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        assert abs(qnorm - 1.0) < 1e-4, (
            f"images.txt: quaternion norm {qnorm:.6f} ≠ 1 for IMAGE_ID {img_id}"
        )

        cam_id_ref = int(parts[8])
        assert cam_id_ref in cameras, (
            f"images.txt: IMAGE_ID {img_id} references unknown CAMERA_ID {cam_id_ref}"
        )

        image_count_actual += 1
        idx += 1  # advance to POINTS2D line

        # Consume line 2 (POINTS2D — must exist, may be blank).
        assert idx < len(lines), (
            f"images.txt: missing POINTS2D line after IMAGE_ID {img_id}"
        )
        idx += 1  # consume POINTS2D line

    if declared_img_count is not None:
        assert declared_img_count == image_count_actual, (
            f"images.txt: header says {declared_img_count} images "
            f"but found {image_count_actual}"
        )

    # ---- points3D.txt: just confirm the file exists and is readable ----
    _ = points3d_path.read_text(encoding="utf-8")

    logger.info(
        "validate_colmap_txt_model: OK — %d camera(s), %d image(s)",
        len(cameras), image_count_actual,
    )


# ---------------------------------------------------------------------------
# Convenience: build T_world_camera from Spot SDK proto objects
# ---------------------------------------------------------------------------

def build_T_world_camera_from_spot(
    snapshot,
    camera_name: str,
    calibration,
) -> np.ndarray:
    """
    Compose the camera-to-world 4×4 transform from Spot SDK objects.

    Parameters
    ----------
    snapshot : bosdyn.api.FrameTreeSnapshot
        Vision-frame snapshot from a ``GetImageResponse`` or ``RobotState``.
        Contains ``child_to_parent_edge_map`` with all frame transforms.
    camera_name : str
        The Spot frame name for this camera sensor, e.g.
        ``"frontleft_fisheye"`` (without the ``"_image"`` suffix used in
        the image service — check your SDK version).
    calibration : bosdyn.api.spot.SpotCameraCalibration
        Calibration proto containing ``body_tform_camera.position`` and
        ``body_tform_camera.rotation`` (body←camera SE3 pose).

    Returns
    -------
    ndarray, shape (4, 4)
        Camera-to-world transform in Spot's vision frame (X-forward / Y-left
        / Z-up).  Pass this directly to ``ColmapWriter.add_image()`` with
        ``frame_R=SPOT_FLU_TO_COLMAP_RDF``.

    Notes
    -----
    This is illustrative pseudo-code: the exact attribute paths depend on
    which version of the BD SDK you are using.  Adjust accordingly.
    """
    from bosdyn.client.frame_helpers import get_se3_a_tform_b, VISION_FRAME_NAME  # type: ignore

    # T_vision_body: world (vision) ← body frame
    T_vision_body_proto = get_se3_a_tform_b(
        snapshot.child_to_parent_edge_map,
        VISION_FRAME_NAME,
        "body",
    )

    def proto_se3_to_matrix(se3_pose) -> np.ndarray:
        p = se3_pose.position
        q = se3_pose.rotation  # w, x, y, z in bosdyn proto
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    T_vision_body = proto_se3_to_matrix(T_vision_body_proto)

    # T_body_camera: body ← camera frame (from calibration proto)
    T_body_camera = proto_se3_to_matrix(calibration.body_tform_camera)

    # T_vision_camera: world ← camera  (camera-to-world)
    return T_vision_body @ T_body_camera


# ---------------------------------------------------------------------------
# Self-contained smoke-test   (python colmap_writer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        sparse_dir = Path(tmp) / "sparse" / "0"

        # ------------------------------------------------------------------
        # Build a 5-camera Spot-like model with OPENCV_FISHEYE
        # ------------------------------------------------------------------
        spot_intrinsics = {
            "back":       {"fx": 330.73, "fy": 331.12, "cx": 311.38, "cy": 241.71,
                           "k1": -0.013, "k2": 0.0007, "k3": -3e-5, "k4": 0.0,
                           "w": 640, "h": 480},
            "frontleft":  {"fx": 329.64, "fy": 330.09, "cx": 240.40, "cy": 320.01,
                           "k1": -0.013, "k2": 0.0007, "k3": -3e-5, "k4": 0.0,
                           "w": 480, "h": 640},
            "frontright": {"fx": 330.04, "fy": 330.31, "cx": 235.20, "cy": 318.79,
                           "k1": -0.013, "k2": 0.0007, "k3": -3e-5, "k4": 0.0,
                           "w": 480, "h": 640},
            "left":       {"fx": 328.21, "fy": 328.56, "cx": 315.70, "cy": 238.28,
                           "k1": -0.013, "k2": 0.0007, "k3": -3e-5, "k4": 0.0,
                           "w": 640, "h": 480},
            "right":      {"fx": 330.20, "fy": 330.42, "cx": 321.98, "cy": 241.27,
                           "k1": -0.013, "k2": 0.0007, "k3": -3e-5, "k4": 0.0,
                           "w": 640, "h": 480},
        }

        # Construct writer with Spot frame rotation baked in.
        with ColmapWriter(
            sparse_dir,
            default_model="OPENCV_FISHEYE",
            default_frame_R=SPOT_FLU_TO_COLMAP_RDF,
        ) as w:

            # Register all five Spot cameras.
            cam_ids = {}
            for name, cfg in spot_intrinsics.items():
                cam_ids[name] = w.register_camera(
                    f"{name}_fisheye_image",
                    width=cfg["w"],
                    height=cfg["h"],
                    intrinsics={k: v for k, v in cfg.items() if k not in ("w", "h")},
                )

            # Simulate 2 timestamps with the robot translated +0.5 m along X.
            for ts, x_offset in enumerate([0.0, 0.5]):
                for cam_name, cam_id in cam_ids.items():
                    # Synthetic cam-to-world: identity rotation, translated.
                    c2w = np.eye(4)
                    c2w[0, 3] = x_offset  # robot moves along Spot X (forward)
                    w.add_image(
                        f"{cam_name}_fisheye_image/{ts:05d}.jpg",
                        cam_id,
                        c2w,
                        # frame_R not needed here — baked into default_frame_R
                    )

        # ------------------------------------------------------------------
        # Print generated files
        # ------------------------------------------------------------------
        for fname in ("cameras.txt", "images.txt", "points3D.txt"):
            fpath = sparse_dir / fname
            print(f"\n{'=' * 62}")
            print(f"  {fname}")
            print(f"{'=' * 62}")
            print(fpath.read_text())

        # ------------------------------------------------------------------
        # Round-trip validation
        # ------------------------------------------------------------------
        validate_colmap_txt_model(sparse_dir)
        print("validate_colmap_txt_model: PASSED")

        # ------------------------------------------------------------------
        # Guard checks
        # ------------------------------------------------------------------
        # Double-save.
        w2 = ColmapWriter(sparse_dir / "new")
        w2.save()
        try:
            w2.save()
            print("ERROR: double-save should have raised RuntimeError", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as e:
            print(f"Double-save guard: OK — {e}")

        # Bad camera_id.
        w3 = ColmapWriter(sparse_dir / "bad_cam")
        try:
            w3.add_image("x.jpg", 999, np.eye(4))
            print("ERROR: bad camera_id should raise ValueError", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Bad camera_id guard: OK — {e}")

        # Bad rotation matrix.
        bad_mat = np.eye(4)
        bad_mat[:3, :3] *= 2.0
        try:
            matrix_to_colmap_pose(bad_mat)
            print("ERROR: bad rotation should raise ValueError", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Bad rotation guard: OK — {e}")

        # Unsupported camera model.
        w4 = ColmapWriter(sparse_dir / "bad_model")
        try:
            w4.register_camera("x", 640, 480, {"fx": 300, "fy": 300}, model="FISHEYE")
            print("ERROR: bad model should raise ValueError", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Bad model guard: OK — {e}")

        # Duplicate image name.
        w5 = ColmapWriter(sparse_dir / "dup")
        cam = w5.register_camera("cam", 640, 480, {"fx": 330, "fy": 330})
        w5.add_image("frame.jpg", cam, np.eye(4))
        try:
            w5.add_image("frame.jpg", cam, np.eye(4))
            print("ERROR: duplicate image should raise ValueError", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Duplicate image guard: OK — {e}")

        # Idempotent register_camera.
        w6 = ColmapWriter(sparse_dir / "idem")
        id_a = w6.register_camera("cam", 640, 480, {"fx": 330, "fy": 330})
        id_b = w6.register_camera("cam", 640, 480, {"fx": 330, "fy": 330})
        assert id_a == id_b, "Idempotency check failed"
        print(f"Idempotent register_camera: OK — ID={id_a}")

        print("\nAll checks passed.")