"""
spot_colmap.py — COLMAP sparse model writer for Boston Dynamics Spot fisheye cameras.

Writes cameras.txt / images.txt / points3D.txt in COLMAP text format.

Key conventions
---------------
Camera model  : OPENCV_FISHEYE  (Kannala-Brandt equidistant, 8 params)
                  fx fy cx cy k1 k2 k3 k4
                  Same parameter order as COLMAP's models.h.

Quaternion    : Hamilton convention, world-to-camera, scalar first
                  QW QX QY QZ
                  scipy.as_quat() returns (x,y,z,w) — we unpack accordingly.

Pose input    : 4×4 cam-to-world matrix in Spot's vision frame
                  (X-forward, Y-left, Z-up, right-handed).
                  cam_to_world[:3, 3] is the camera centre in world coords.
                  The writer inverts this to world-to-camera for COLMAP.

In-plane rot  : When a camera is physically mounted rotated (e.g. frontleft /
                  frontright are mounted 90° CCW), the caller must pass the
                  rotation-compensated cam_to_world.  See apply_inplane_rotation().

World frame   : Spot's vision frame is used as-is.  It is a valid right-handed
                  frame for COLMAP — the scene will appear "on its side" in the
                  default COLMAP view (because Spot uses X-forward/Z-up rather
                  than Y-up), but camera relative poses are correct.  Use COLMAP's
                  view controls to rotate the scene if desired.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-plane rotation helper
# ---------------------------------------------------------------------------

def apply_inplane_rotation(cam_to_world: np.ndarray, rotation_deg: float) -> np.ndarray:
    """
    Compose an in-plane camera-frame rotation into a cam-to-world matrix.

    Call this when the image pixels were rotated by ``ndimage.rotate(img, rotation_deg)``
    so that the COLMAP extrinsic and the displayed image agree on which direction
    is "right" and "down" in the camera frame.

    Derivation
    ----------
    When ndimage.rotate(img, angle) rotates the image by ``angle`` degrees
    counter-clockwise, it remaps pixel coordinates as::

        x_display = cos(angle)*x_orig - sin(angle)*y_orig
        y_display = sin(angle)*x_orig + cos(angle)*y_orig

    To express the original camera axes in terms of the display camera axes we
    need the inverse (transpose) rotation.  The cam-to-world for the *display*
    camera frame is::

        cam_to_world_display = cam_to_world_original @ Rz(-angle)

    Parameters
    ----------
    cam_to_world : ndarray, shape (4, 4)
        Camera-to-world transform in the original (SDK-reported) sensor frame.
    rotation_deg : float
        Image rotation that was applied for display (degrees, positive = CCW).
        Typical Spot values:  frontleft = -90,  frontright = -90,  right = 180.

    Returns
    -------
    ndarray, shape (4, 4)
        Camera-to-world transform in the display-corrected camera frame.
    """
    if rotation_deg == 0:
        return cam_to_world

    theta = np.radians(-rotation_deg)   # negate: we undo the image rotation
    c, s = np.cos(theta), np.sin(theta)

    # 4×4 rotation around camera Z axis (optical axis)
    R_z = np.array(
        [[c, -s, 0.0, 0.0],
         [s,  c, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cam_to_world @ R_z


# ---------------------------------------------------------------------------
# Pose conversion
# ---------------------------------------------------------------------------

def cam_to_world_to_colmap_pose(
    cam_to_world: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Convert a 4×4 cam-to-world matrix to COLMAP extrinsic parameters.

    Returns
    -------
    (qw, qx, qy, qz, tx, ty, tz)
        COLMAP world-to-camera extrinsic.  Quaternion is Hamilton convention,
        scalar (w) first.
    """
    R_c2w = cam_to_world[:3, :3]
    t_c2w = cam_to_world[:3, 3]

    # Exact closed-form rigid-body inverse (avoids LU accumulation errors).
    R_wc = R_c2w.T
    t_wc = -R_wc @ t_c2w

    # scipy as_quat() → (x, y, z, w); COLMAP wants (w, x, y, z).
    qx, qy, qz, qw = Rotation.from_matrix(R_wc).as_quat()

    return (
        float(qw), float(qx), float(qy), float(qz),
        float(t_wc[0]), float(t_wc[1]), float(t_wc[2]),
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class SpotColmapWriter:
    """
    Buffer Spot camera intrinsics and image poses, then flush to a COLMAP
    text-format sparse model on :meth:`save`.

    Interface is intentionally compatible with the old ``ColmapWriter`` so
    ``spot_controller.py`` only needs a one-line import change.

    Usage
    -----
    ::

        writer = SpotColmapWriter(sparse_dir)

        cam_id = writer.register_camera(
            "back_fisheye_image",
            width=640, height=480,
            intrinsics={"fx": 330, "fy": 330, "cx": 320, "cy": 240,
                        "k1": -0.009, "k2": 0.00046, "k3": -0.000019, "k4": 0.0},
        )

        corrected = apply_inplane_rotation(cam_to_world, rotation_deg=0)
        writer.add_image("back_fisheye_image/00000.jpg", cam_id, corrected)

        writer.save()
    """

    def __init__(self, sparse_dir) -> None:
        self._dir = Path(sparse_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._cameras: Dict[str, dict] = {}   # source_name → record
        self._images:  List[dict] = []
        self._names_seen: set = set()

        self._next_cam_id = 1
        self._next_img_id = 1
        self._saved = False

    # ---------------------------------------------------------------- cameras

    def register_camera(
        self,
        source_name: str,
        width: int,
        height: int,
        intrinsics: dict,
    ) -> int:
        """
        Register a camera and return its COLMAP CAMERA_ID.

        Idempotent: repeated calls with the same *source_name* return the
        existing ID without modifying any state.

        Parameters
        ----------
        source_name : str
            Human-readable label, e.g. ``"frontleft_fisheye_image"``.
        width, height : int
            Sensor dimensions AFTER any display rotation (i.e. the dimensions
            of the image actually saved to disk).
        intrinsics : dict
            Keys: ``fx``, ``fy`` (required); ``cx``, ``cy`` (default: image
            centre); ``k1``, ``k2``, ``k3``, ``k4`` (default: 0.0).
        """
        if source_name in self._cameras:
            existing_id = self._cameras[source_name]['id']
            logger.debug("register_camera: '%s' already registered as id=%d", source_name, existing_id)
            return existing_id

        if 'fx' not in intrinsics or 'fy' not in intrinsics:
            raise ValueError(f"register_camera: 'fx' and 'fy' are required in intrinsics for '{source_name}'")

        cam_id = self._next_cam_id
        self._next_cam_id += 1

        fx  = float(intrinsics['fx'])
        fy  = float(intrinsics['fy'])
        cx  = float(intrinsics.get('cx', width  / 2.0))
        cy  = float(intrinsics.get('cy', height / 2.0))
        k1  = float(intrinsics.get('k1', 0.0))
        k2  = float(intrinsics.get('k2', 0.0))
        k3  = float(intrinsics.get('k3', 0.0))
        k4  = float(intrinsics.get('k4', 0.0))

        self._cameras[source_name] = {
            'id':     cam_id,
            'width':  int(width),
            'height': int(height),
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4,
        }
        logger.info(
            "Registered camera %d: %-35s  %dx%d  "
            "fx=%.4g fy=%.4g cx=%.4g cy=%.4g  "
            "k=[%.5g %.5g %.5g %.5g]",
            cam_id, source_name, width, height,
            fx, fy, cx, cy, k1, k2, k3, k4,
        )
        return cam_id

    # ----------------------------------------------------------------- images

    def add_image(
        self,
        image_name: str,
        camera_id: int,
        cam_to_world: np.ndarray,
    ) -> int:
        """
        Buffer one image entry.

        Parameters
        ----------
        image_name : str
            Path relative to the COLMAP images folder, e.g.
            ``"back_fisheye_image/00042.jpg"``.  Must be unique.
        camera_id : int
            COLMAP CAMERA_ID returned by :meth:`register_camera`.
        cam_to_world : ndarray, shape (4, 4)
            Camera-to-world rigid transform.  Pass the output of
            :func:`apply_inplane_rotation` when the image has been display-rotated.

        Returns
        -------
        int
            Assigned IMAGE_ID, or -1 if *image_name* is a duplicate (logged
            as a warning; not raised as an exception to keep recording alive).
        """
        if image_name in self._names_seen:
            logger.warning("Skipping duplicate image_name: '%s'", image_name)
            return -1

        # Sanity-check the rotation block.
        R = cam_to_world[:3, :3]
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.02:
            logger.warning(
                "add_image '%s': rotation block det=%.5f (expected 1.0) — "
                "pose may be invalid",
                image_name, det,
            )

        qw, qx, qy, qz, tx, ty, tz = cam_to_world_to_colmap_pose(cam_to_world)

        img_id = self._next_img_id
        self._next_img_id += 1
        self._names_seen.add(image_name)
        self._images.append({
            'id':        img_id,
            'name':      image_name,
            'camera_id': camera_id,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
        })

        # Log camera centre (always the same as cam_to_world[:3, 3]).
        cc = cam_to_world[:3, 3]
        logger.debug(
            "Buffered image %d: %-50s  cam_id=%d  "
            "centre=(%.4f, %.4f, %.4f)  q=(%.4f %.4f %.4f %.4f)",
            img_id, image_name, camera_id,
            cc[0], cc[1], cc[2],
            qw, qx, qy, qz,
        )
        return img_id

    # Alias for backward compatibility
    def write_image(self, image_name: str, camera_id: int, cam_to_world: np.ndarray) -> int:
        return self.add_image(image_name, camera_id, cam_to_world)

    # ------------------------------------------------------------------ save

    def save(self) -> None:
        """Write cameras.txt, images.txt, and points3D.txt."""
        if self._saved:
            logger.warning(
                "SpotColmapWriter.save() already called; re-saving to %s", self._dir
            )
        self._write_cameras_txt()
        self._write_images_txt()
        self._write_points3d_txt()
        self._saved = True
        logger.info(
            "COLMAP sparse model saved → %s  (%d camera(s), %d image(s))",
            self._dir, len(self._cameras), len(self._images),
        )

    def reset(self) -> None:
        """Clear all buffered state so the writer can be reused for a new session."""
        self._cameras.clear()
        self._images.clear()
        self._names_seen.clear()
        self._next_cam_id = 1
        self._next_img_id = 1
        self._saved = False
        logger.debug("SpotColmapWriter reset — ready for new session.")

    # ---------------------------------------------------------------- queries

    @property
    def num_cameras(self) -> int:
        return len(self._cameras)

    @property
    def num_images(self) -> int:
        return len(self._images)

    # ---------------------------------------------------------- file writers

    def _write_cameras_txt(self) -> None:
        lines: List[str] = [
            "# Camera list with one line of data per camera:\n",
            "#   CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n",
            "#   OPENCV_FISHEYE params: fx fy cx cy k1 k2 k3 k4\n",
            f"# Number of cameras: {len(self._cameras)}\n",
        ]
        for cam in self._cameras.values():
            lines.append(
                f"{cam['id']} OPENCV_FISHEYE {cam['width']} {cam['height']} "
                f"{cam['fx']:.10g} {cam['fy']:.10g} "
                f"{cam['cx']:.10g} {cam['cy']:.10g} "
                f"{cam['k1']:.10g} {cam['k2']:.10g} "
                f"{cam['k3']:.10g} {cam['k4']:.10g}\n"
            )
        (self._dir / "cameras.txt").write_text("".join(lines), encoding="utf-8")

    def _write_images_txt(self) -> None:
        lines: List[str] = [
            "# Image list with two lines of data per image:\n",
            "#   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
            f"# Number of images: {len(self._images)}, mean observations per image: 0\n",
        ]
        for img in self._images:
            lines.append(
                f"{img['id']} "
                f"{img['qw']:.10g} {img['qx']:.10g} "
                f"{img['qy']:.10g} {img['qz']:.10g} "
                f"{img['tx']:.10g} {img['ty']:.10g} {img['tz']:.10g} "
                f"{img['camera_id']} {img['name']}\n"
            )
            lines.append("\n")   # required empty POINTS2D line
        (self._dir / "images.txt").write_text("".join(lines), encoding="utf-8")

    def _write_points3d_txt(self) -> None:
        (self._dir / "points3D.txt").write_text(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0, mean track length: 0\n",
            encoding="utf-8",
        )
