"""
colmap_writer.py
================
Writes a COLMAP text-format sparse model (cameras.txt / images.txt / points3D.txt)
from externally supplied camera intrinsics and cam-to-world poses.

Format reference
----------------
https://colmap.github.io/legacy/3.9/format.html  (v3.9, the stable spec)
https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h  (PINHOLE params)
https://github.com/colmap/colmap/blob/1a4d0bad2e90aa65ce997c9d1779518eaed998d5/scripts/python/read_write_model.py

Key conventions (verified against source / docs)
-------------------------------------------------
- cameras.txt  : one line per camera — CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
                 PINHOLE params order: fx fy cx cy    (verified in models.h)
- images.txt   : TWO lines per image
                 line 1 — IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                 line 2 — POINTS2D[] as (X Y POINT3D_ID); empty for prior-pose import
- points3D.txt : empty (one header only); COLMAP fills this during reconstruction
- Quaternion   : Hamilton convention, world-to-camera, (QW QX QY QZ) scalar first
                 scipy.spatial.transform.Rotation.as_quat() returns (x,y,z,w),
                 so we unpack as qx,qy,qz,qw and write qw first.
- IDs          : CAMERA_ID and IMAGE_ID start at 1 (1-based), per COLMAP convention
                 (COLMAP's own database uses 1-based IDs and the text format mirrors this)
- Header stats : Every file's 3rd comment line is the official count/stats summary.
                 COLMAP's reader ignores comment lines, but downstream tools (pycolmap,
                 nerfstudio, colmap2nerf) parse these summary lines — they must be correct.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quaternion / pose helpers
# ---------------------------------------------------------------------------

def _validate_rotation_matrix(R: np.ndarray, *, label: str = "R") -> None:
    """
    Raise ValueError if R is not a valid 3×3 proper rotation matrix.

    Checks:
      - shape is (3, 3)
      - no NaN / Inf entries
      - columns are orthonormal  (R^T R ≈ I,  tol = 1e-5)
      - determinant is +1        (proper rotation, not a reflection)
    """
    if R.shape != (3, 3):
        raise ValueError(f"{label}: expected shape (3,3), got {R.shape}")
    if not np.isfinite(R).all():
        raise ValueError(f"{label}: contains non-finite values (NaN or Inf)")
    eye_diff = np.max(np.abs(R.T @ R - np.eye(3)))
    if eye_diff > 1e-5:
        raise ValueError(
            f"{label}: columns are not orthonormal — max deviation from I: {eye_diff:.2e}"
        )
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-5:
        raise ValueError(
            f"{label}: determinant is {det:.6f} (expected +1.0 for a proper rotation)"
        )


def _validate_cam_to_world(cam_to_world: np.ndarray) -> None:
    """
    Raise ValueError if cam_to_world is not a valid 4×4 rigid-body transform.
    """
    if cam_to_world.shape != (4, 4):
        raise ValueError(
            f"cam_to_world: expected shape (4,4), got {cam_to_world.shape}"
        )
    if not np.isfinite(cam_to_world).all():
        raise ValueError("cam_to_world: contains non-finite values (NaN or Inf)")
    # Bottom row must be [0, 0, 0, 1]
    bottom = cam_to_world[3, :]
    if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(
            f"cam_to_world: bottom row must be [0,0,0,1], got {bottom}"
        )
    # Validate rotation sub-block
    _validate_rotation_matrix(cam_to_world[:3, :3], label="cam_to_world[:3,:3]")


def matrix_to_colmap_pose(
    cam_to_world: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Convert a 4×4 camera-to-world matrix to COLMAP's extrinsic representation.

    COLMAP stores the *world-to-camera* transform:
        X_cam = R @ X_world + t

    as a 7-tuple (QW, QX, QY, QZ, TX, TY, TZ) using Hamilton quaternion convention
    with the scalar part (w) written first — matching the images.txt column order.

    Parameters
    ----------
    cam_to_world : np.ndarray, shape (4, 4)
        Camera-to-world rigid transform (rotation + translation).
        The camera centre in world coordinates is cam_to_world[:3, 3].

    Returns
    -------
    (qw, qx, qy, qz, tx, ty, tz) : tuple of float
        COLMAP extrinsic parameters (world-to-camera).

    Raises
    ------
    ValueError
        If cam_to_world is not a valid 4×4 rigid-body transform.

    Notes
    -----
    scipy's Rotation.as_quat() returns (x, y, z, w) — the Hamilton convention
    with the scalar part *last*.  We unpack accordingly and then reorder to
    produce (w, x, y, z) = (QW, QX, QY, QZ) as COLMAP expects.

    The camera centre expressed in world coordinates is:
        C_world = -R^T @ t
    which equals cam_to_world[:3, 3] by construction.
    """
    _validate_cam_to_world(cam_to_world)

    world_to_cam = np.linalg.inv(cam_to_world)
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]

    # scipy returns (x, y, z, w); unpack into named variables for clarity
    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()  # noqa: unpack order matches scipy

    return (
        float(qw), float(qx), float(qy), float(qz),
        float(t[0]), float(t[1]), float(t[2]),
    )


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class _CameraRecord:
    """Internal storage for one registered camera."""

    __slots__ = ("camera_id", "source_name", "width", "height", "fx", "fy", "cx", "cy")

    def __init__(
        self,
        camera_id: int,
        source_name: str,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> None:
        self.camera_id = camera_id
        self.source_name = source_name
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


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

    Only the three text files required for COLMAP import are produced:

    * ``cameras.txt``   — PINHOLE intrinsics
    * ``images.txt``    — world-to-camera extrinsics (no 2D keypoints)
    * ``points3D.txt``  — empty; COLMAP fills this during reconstruction

    File format
    -----------
    Strictly follows the official specification at
    https://colmap.github.io/legacy/3.9/format.html
    and matches the output of COLMAP's own ``model_converter --output_type TXT``.

    Usage
    -----
    ::

        writer = ColmapWriter(sparse_dir)
        cam_id = writer.register_camera(
            "cam0", width=1920, height=1080,
            intrinsics={"fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0},
        )
        writer.add_image("frame_0001.jpg", cam_id, cam_to_world_4x4)
        writer.save()

    Context-manager form (calls :meth:`save` automatically)::

        with ColmapWriter(sparse_dir) as w:
            cam_id = w.register_camera(...)
            w.add_image(...)

    Parameters
    ----------
    sparse_dir : path-like
        Destination directory.  Created (including parents) if absent.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, sparse_dir) -> None:
        self._sparse_dir = Path(sparse_dir)
        self._sparse_dir.mkdir(parents=True, exist_ok=True)

        # source_name → _CameraRecord
        self._cameras: Dict[str, _CameraRecord] = {}
        self._next_camera_id: int = 1

        # List of _ImageRecord, in insertion order
        self._images: List[_ImageRecord] = []
        self._next_image_id: int = 1

        # Guard against double-save
        self._saved: bool = False

        # Track registered image names to detect duplicates
        self._image_names_seen: Set[str] = set()

    # --------------------------------------------------------------- cameras

    def register_camera(
        self,
        source_name: str,
        width: int,
        height: int,
        intrinsics: dict,
    ) -> int:
        """
        Register a PINHOLE camera and return its COLMAP ``CAMERA_ID``.

        Calling this method again with the same *source_name* is idempotent —
        the existing ``CAMERA_ID`` is returned without modifying any state.

        Parameters
        ----------
        source_name : str
            Arbitrary label identifying this camera's configuration (e.g.,
            ``"rgb"`` or ``"frontleft_fisheye"``).  Multiple images can share
            the same camera by passing the returned ``CAMERA_ID`` to
            :meth:`add_image`.
        width, height : int
            Sensor dimensions in pixels.  Must be positive.
        intrinsics : dict
            Camera intrinsic parameters.  Keys used:

            * ``"fx"``  — horizontal focal length in pixels (required, > 0)
            * ``"fy"``  — vertical focal length in pixels (required, > 0)
            * ``"cx"``  — principal-point x in pixels (optional; default width/2)
            * ``"cy"``  — principal-point y in pixels (optional; default height/2)

        Returns
        -------
        int
            Assigned COLMAP ``CAMERA_ID`` (1-based integer).

        Raises
        ------
        TypeError
            If *source_name* is not a str, or dimensions are not integers.
        ValueError
            If dimensions are non-positive, or ``fx``/``fy`` are missing or
            non-positive.
        """
        if not isinstance(source_name, str):
            raise TypeError(f"source_name must be str, got {type(source_name).__name__}")
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("width and height must be integers")
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got {width}×{height}")

        # Idempotent registration
        if source_name in self._cameras:
            existing = self._cameras[source_name]
            logger.debug(
                "register_camera: '%s' already registered as camera_id=%d",
                source_name, existing.camera_id,
            )
            return existing.camera_id

        # Validate and extract focal lengths (required)
        fx = intrinsics.get("fx")
        fy = intrinsics.get("fy")
        if fx is None or fy is None:
            raise ValueError(
                f"intrinsics for '{source_name}' must contain 'fx' and 'fy'"
            )
        fx, fy = float(fx), float(fy)
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(
                f"fx and fy must be positive; got fx={fx}, fy={fy}"
            )

        # Principal point (optional; default to image centre per COLMAP convention)
        cx = float(intrinsics.get("cx", width / 2.0))
        cy = float(intrinsics.get("cy", height / 2.0))

        camera_id = self._next_camera_id
        self._next_camera_id += 1
        self._cameras[source_name] = _CameraRecord(
            camera_id=camera_id,
            source_name=source_name,
            width=width,
            height=height,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
        logger.debug(
            "Registered camera %d: '%s' %dx%d  fx=%.4f fy=%.4f cx=%.4f cy=%.4f",
            camera_id, source_name, width, height, fx, fy, cx, cy,
        )
        return camera_id

    # ----------------------------------------------------------------- images

    def add_image(
        self,
        image_name: str,
        camera_id: int,
        cam_to_world: np.ndarray,
    ) -> int:
        """
        Buffer one image entry for writing.

        Parameters
        ----------
        image_name : str
            Path relative to the COLMAP images folder, e.g.
            ``"frontleft_fisheye_image/00042.jpg"``.
            Must be unique across all buffered images.
        camera_id : int
            COLMAP ``CAMERA_ID`` returned by :meth:`register_camera`.
        cam_to_world : np.ndarray, shape (4, 4)
            Camera-to-world rigid transform.
            ``cam_to_world[:3, 3]`` is the camera centre in world coordinates.

        Returns
        -------
        int
            Assigned COLMAP ``IMAGE_ID`` (1-based integer).

        Raises
        ------
        ValueError
            If *camera_id* has not been registered, *image_name* is a
            duplicate, or *cam_to_world* is not a valid rigid transform.
        """
        # Validate camera_id
        registered_ids = {rec.camera_id for rec in self._cameras.values()}
        if camera_id not in registered_ids:
            raise ValueError(
                f"camera_id={camera_id} has not been registered. "
                f"Registered IDs: {sorted(registered_ids)}"
            )

        # Validate image_name uniqueness
        if image_name in self._image_names_seen:
            raise ValueError(
                f"Duplicate image_name: '{image_name}' has already been added"
            )

        # Convert pose (also validates cam_to_world internally)
        qw, qx, qy, qz, tx, ty, tz = matrix_to_colmap_pose(cam_to_world)

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
        logger.debug("Buffered image %d: '%s' (camera_id=%d)", image_id, image_name, camera_id)
        return image_id

    # NOTE: keep the original method name as an alias so existing call-sites
    # that used write_image() continue to work without changes.
    def write_image(
        self,
        image_name: str,
        camera_id: int,
        cam_to_world: np.ndarray,
    ) -> int:
        """Alias for :meth:`add_image` (backward compatibility)."""
        return self.add_image(image_name, camera_id, cam_to_world)

    # ------------------------------------------------------------------ save

    def save(self) -> None:
        """
        Write ``cameras.txt``, ``images.txt``, and ``points3D.txt`` to the
        directory supplied at construction time.

        Calling :meth:`save` more than once raises :exc:`RuntimeError` to
        prevent accidental overwrites.

        Raises
        ------
        RuntimeError
            If :meth:`save` has already been called on this instance.
        """
        if self._saved:
            raise RuntimeError(
                "ColmapWriter.save() has already been called on this instance. "
                "Create a new ColmapWriter to write another model."
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
        for a new capture session without constructing a new instance.

        The destination directory is preserved.  The next :meth:`save` call
        will overwrite any files previously written there.

        Typical use: call ``reset()`` immediately after :meth:`save` so the
        writer is ready for the next route or manual-walk session.
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
        # Only save if no exception was raised; this avoids writing a partial
        # model when the caller's body raises.
        if exc_type is None:
            self.save()
        return False  # do not suppress exceptions

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
        Write ``cameras.txt`` in the official COLMAP text format.

        Format (one line per camera)::

            # Camera list with one line of data per camera:
            #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            # Number of cameras: N
            CAMERA_ID PINHOLE WIDTH HEIGHT fx fy cx cy

        PINHOLE parameter order (from models.h):
            params[0]=fx, params[1]=fy, params[2]=cx, params[3]=cy
        """
        # Official 3rd header line: "# Number of cameras: N"
        header = (
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"# Number of cameras: {len(self._cameras)}\n"
        )

        lines = [header]
        for rec in self._cameras.values():
            # PINHOLE params order (verified in models.h): fx fy cx cy
            line = (
                f"{rec.camera_id} PINHOLE {rec.width} {rec.height} "
                f"{rec.fx:.10g} {rec.fy:.10g} {rec.cx:.10g} {rec.cy:.10g}\n"
            )
            lines.append(line)

        (self._sparse_dir / "cameras.txt").write_text("".join(lines), encoding="utf-8")

    def _write_images_txt(self) -> None:
        """
        Write ``images.txt`` in the official COLMAP text format.

        Format (TWO lines per image)::

            # Image list with two lines of data per image:
            #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            #   POINTS2D[] as (X, Y, POINT3D_ID)
            # Number of images: N, mean observations per image: 0
            IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            <empty POINTS2D line>

        Notes
        -----
        * Quaternion convention: Hamilton, world-to-camera, scalar (w) first.
        * The POINTS2D line is intentionally empty because no keypoint
          observations are available in a prior-pose import workflow.
          COLMAP accepts an empty line here and will re-extract features
          from the images during feature extraction.
        * Floating-point precision: 10 significant digits (matching
          ``%.10g`` format) — more than COLMAP's own C++ output (6 sig-figs)
          but harmless and avoids rounding loss for large coordinates.
        """
        # Official 3rd header line: "# Number of images: N, mean observations per image: 0"
        header = (
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {len(self._images)}, mean observations per image: 0\n"
        )

        lines = [header]
        for rec in self._images:
            # Line 1: pose + metadata
            pose_line = (
                f"{rec.image_id} "
                f"{rec.qw:.10g} {rec.qx:.10g} {rec.qy:.10g} {rec.qz:.10g} "
                f"{rec.tx:.10g} {rec.ty:.10g} {rec.tz:.10g} "
                f"{rec.camera_id} {rec.image_name}\n"
            )
            # Line 2: empty POINTS2D list (required by the two-line-per-image format)
            lines.append(pose_line)
            lines.append("\n")

        (self._sparse_dir / "images.txt").write_text("".join(lines), encoding="utf-8")

    def _write_points3d_txt(self) -> None:
        """
        Write an empty ``points3D.txt`` in the official COLMAP text format.

        Format::

            # 3D point list with one line of data per point:
            #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            # Number of points: 0, mean track length: 0

        The file intentionally contains no data lines.  COLMAP populates
        this file during feature matching and triangulation.
        """
        content = (
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0, mean track length: 0\n"
        )
        (self._sparse_dir / "points3D.txt").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation helper (post-write round-trip check)
# ---------------------------------------------------------------------------

def validate_colmap_txt_model(sparse_dir) -> None:
    """
    Parse the three COLMAP text files just written and assert internal
    consistency.  Intended for use in tests and CI pipelines.

    Checks performed
    ----------------
    * All three files exist and are non-empty (except points3D.txt, which is
      expected to be empty of data lines).
    * Every CAMERA_ID referenced in images.txt exists in cameras.txt.
    * Every IMAGE_ID is unique.
    * Every CAMERA_ID is unique.
    * Quaternion norm ≈ 1 for every image.
    * cameras.txt count header matches actual camera count.
    * images.txt count header matches actual image count.

    Parameters
    ----------
    sparse_dir : path-like
        Directory containing ``cameras.txt``, ``images.txt``,
        ``points3D.txt``.

    Raises
    ------
    AssertionError
        On the first consistency violation found.
    FileNotFoundError
        If any of the three required files is absent.
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
            # Extract "# Number of cameras: N"
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
        assert model == "PINHOLE", (
            f"cameras.txt: unexpected model '{model}' — only PINHOLE is written by ColmapWriter"
        )
        width, height = int(parts[2]), int(parts[3])
        assert width > 0 and height > 0, f"cameras.txt: non-positive dimensions for cam {cam_id}"
        params = [float(p) for p in parts[4:]]
        assert len(params) == 4, (
            f"cameras.txt: PINHOLE expects 4 params (fx fy cx cy), got {len(params)} for cam {cam_id}"
        )
        fx, fy, cx, cy = params
        assert fx > 0.0 and fy > 0.0, f"cameras.txt: non-positive focal length for cam {cam_id}"
        cameras[cam_id] = {"model": model, "width": width, "height": height,
                           "fx": fx, "fy": fy, "cx": cx, "cy": cy}

    if declared_cam_count is not None:
        assert declared_cam_count == len(cameras), (
            f"cameras.txt: header says {declared_cam_count} cameras but found {len(cameras)}"
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
            # Extract "# Number of images: N, mean observations per image: M"
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

        # Line 1 of a two-line image block
        parts = stripped.split()
        assert len(parts) >= 9, f"images.txt: malformed pose line: {line!r}"
        img_id = int(parts[0])
        assert img_id not in image_ids_seen, f"images.txt: duplicate IMAGE_ID {img_id}"
        image_ids_seen.add(img_id)

        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        qnorm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        assert abs(qnorm - 1.0) < 1e-4, (
            f"images.txt: quaternion norm {qnorm:.6f} ≠ 1 for IMAGE_ID {img_id}"
        )

        cam_id_ref = int(parts[8])
        assert cam_id_ref in cameras, (
            f"images.txt: IMAGE_ID {img_id} references unknown CAMERA_ID {cam_id_ref}"
        )

        image_count_actual += 1

        # Consume line 2 (POINTS2D — must exist, may be blank)
        idx += 1
        if idx < len(lines):
            # Just consume the line; it may be empty or contain (X Y POINT3D_ID) triples
            idx += 1
        else:
            assert False, f"images.txt: missing POINTS2D line after IMAGE_ID {img_id}"

    if declared_img_count is not None:
        assert declared_img_count == image_count_actual, (
            f"images.txt: header says {declared_img_count} images but found {image_count_actual}"
        )

    # ---- points3D.txt: confirm it has no data lines ----
    for line in points3d_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            # Data lines present — not an error per se (they could be pre-populated),
            # but we log a warning since ColmapWriter always writes an empty file.
            logger.warning(
                "points3D.txt contains data lines — unexpected for a prior-pose import model"
            )
            break

    logger.info(
        "validate_colmap_txt_model: OK — %d camera(s), %d image(s)",
        len(cameras), image_count_actual,
    )


# ---------------------------------------------------------------------------
# Self-contained smoke-test  (python colmap_writer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        sparse_dir = Path(tmp) / "sparse" / "0"

        # ---- build a tiny two-camera, three-image model ----
        with ColmapWriter(sparse_dir) as w:

            # Camera 0: 1920×1080 with square pixels
            cam0 = w.register_camera(
                "rgb_front",
                width=1920, height=1080,
                intrinsics={"fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0},
            )

            # Camera 1: same optics, different resolution
            cam1 = w.register_camera(
                "rgb_rear",
                width=1280, height=720,
                intrinsics={"fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
            )

            # Idempotency: registering the same source_name again returns the same ID
            cam0_again = w.register_camera(
                "rgb_front", width=1920, height=1080,
                intrinsics={"fx": 1200.0, "fy": 1200.0},
            )
            assert cam0 == cam0_again, "Idempotency check failed"

            # Three images with identity rotation, translated along X
            for i, offset in enumerate([0.0, 0.5, 1.0]):
                c2w = np.eye(4)
                c2w[0, 3] = offset          # camera moves along world X
                cid = cam0 if i < 2 else cam1
                w.add_image(f"frame_{i:04d}.jpg", cid, c2w)

        # ---- print what was written ----
        for fname in ("cameras.txt", "images.txt", "points3D.txt"):
            fpath = sparse_dir / fname
            print(f"\n{'='*60}")
            print(f"  {fname}")
            print(f"{'='*60}")
            print(fpath.read_text())

        # ---- run consistency validation ----
        validate_colmap_txt_model(sparse_dir)

        # ---- double-save guard ----
        w2 = ColmapWriter(sparse_dir / "new")
        w2.save()
        try:
            w2.save()
            print("ERROR: double-save should have raised RuntimeError", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as e:
            print(f"Double-save guard works: {e}")

        # ---- bad camera_id reference ----
        w3 = ColmapWriter(sparse_dir / "bad")
        try:
            w3.add_image("x.jpg", 999, np.eye(4))
            print("ERROR: should have raised ValueError", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Bad camera_id guard works: {e}")

        # ---- bad rotation matrix ----
        bad_mat = np.eye(4)
        bad_mat[:3, :3] *= 2.0          # not orthonormal
        try:
            matrix_to_colmap_pose(bad_mat)
            print("ERROR: should have raised ValueError", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Bad rotation guard works: {e}")

        print("\nAll checks passed.")