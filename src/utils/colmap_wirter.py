import struct
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_PINHOLE_MODEL_ID   = 1   # cameras.bin model_id for PINHOLE
_PINHOLE_NUM_PARAMS = 4   # fx fy cx cy


def matrix_to_colmap_pose(
    cam_to_world: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Convert a 4x4 camera-to-world matrix to COLMAP extrinsic convention.

    COLMAP stores the *world-to-camera* transform:
        X_cam = R @ X_world + t
    as (qw, qx, qy, qz, tx, ty, tz) with Hamilton quaternion convention.
    """
    world_to_cam = np.linalg.inv(cam_to_world)
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]
    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()   # scipy → (x,y,z,w)
    return float(qw), float(qx), float(qy), float(qz), float(t[0]), float(t[1]), float(t[2])


class ColmapWriter:
    """
    Buffers camera and image data in memory, then writes either
    COLMAP text (.txt) or binary (.bin) sparse model files on save().

    Usage
    -----
    with ColmapWriter(sparse_dir, fmt="txt") as writer:
        camera_id = writer.register_camera(...)
        writer.write_image(...)
    # save() is called automatically on __exit__

    fmt="txt"  →  cameras.txt / images.txt / points3D.txt
    fmt="bin"  →  cameras.bin / images.bin / points3D.bin
    """

    FORMATS = ("txt", "bin")

    def __init__(self, sparse_dir: Path, fmt: str = "txt") -> None:
        if fmt not in self.FORMATS:
            raise ValueError(f"fmt must be one of {self.FORMATS}, got {fmt!r}")

        self.sparse_dir = Path(sparse_dir)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt

        # source_name → (camera_id, width, height, intrinsics_dict)
        self._cameras: Dict[str, Tuple[int, int, int, dict]] = {}
        self._next_camera_id = 1

        # (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)
        self._images: List[Tuple] = []
        self._next_image_id = 1


    def register_camera(
        self, source_name: str, width: int, height: int, intrinsics: dict
    ) -> int:
        """Idempotent — returns existing camera_id on repeat calls."""
        if source_name in self._cameras:
            return self._cameras[source_name][0]

        camera_id = self._next_camera_id
        self._next_camera_id += 1
        self._cameras[source_name] = (camera_id, width, height, intrinsics)
        logger.debug("Registered camera %d: %s (%dx%d)", camera_id, source_name, width, height)
        return camera_id

    def write_image(
        self, image_name: str, camera_id: int, cam_to_world: np.ndarray
    ) -> int:
        """
        Buffer one image entry.
        image_name is relative to images/, e.g. 'frontleft_fisheye_image/00042.jpg'.
        Returns the assigned COLMAP image_id.
        """
        image_id = self._next_image_id
        self._next_image_id += 1
        self._images.append((image_id, *matrix_to_colmap_pose(cam_to_world), camera_id, image_name))
        logger.debug("Buffered image %d: %s", image_id, image_name)
        return image_id

    def save(self) -> None:
        """Write all buffered data to disk in the configured format."""
        if self.fmt == "txt":
            self._write_cameras_txt()
            self._write_images_txt()
            self._write_points3d_txt()
        else:
            self._write_cameras_bin()
            self._write_images_bin()
            self._write_points3d_bin()

        logger.info(
            "COLMAP model saved [%s] → %s  (%d camera(s), %d image(s))",
            self.fmt, self.sparse_dir, len(self._cameras), len(self._images),
        )


    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.save()

    # ── txt writers ───────────────────────────────────────────────────────────

    def _write_cameras_txt(self) -> None:
        lines = [
            "# Camera list with one line of data per camera:\n",
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
            "# PINHOLE params: fx fy cx cy\n",
        ]
        for _, (camera_id, width, height, intr) in self._cameras.items():
            fx = intr.get("fx", 0.0);  fy = intr.get("fy", 0.0)
            cx = intr.get("cx", width / 2.0);  cy = intr.get("cy", height / 2.0)
            lines.append(
                f"{camera_id} PINHOLE {width} {height} "
                f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
            )
        (self.sparse_dir / "cameras.txt").write_text("".join(lines))

    def _write_images_txt(self) -> None:
        lines = [
            "# Image list with two lines of data per image:\n",
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        ]
        for (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name) in self._images:
            lines.append(
                f"{image_id} "
                f"{qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{tx:.9f} {ty:.9f} {tz:.9f} "
                f"{camera_id} {name}\n"
                "\n"   # empty POINTS2D line
            )
        (self.sparse_dir / "images.txt").write_text("".join(lines))

    def _write_points3d_txt(self) -> None:
        (self.sparse_dir / "points3D.txt").write_text(
            "# 3D point list — empty; populated by COLMAP reconstruction\n"
            "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )

    # ── bin writers ───────────────────────────────────────────────────────────

    def _write_cameras_bin(self) -> None:
        with open(self.sparse_dir / "cameras.bin", "wb") as f:
            f.write(struct.pack("<Q", len(self._cameras)))
            for _, (camera_id, width, height, intr) in self._cameras.items():
                fx = intr.get("fx", 0.0);  fy = intr.get("fy", 0.0)
                cx = intr.get("cx", width / 2.0);  cy = intr.get("cy", height / 2.0)
                f.write(struct.pack("<I",  camera_id))
                f.write(struct.pack("<i",  _PINHOLE_MODEL_ID))
                f.write(struct.pack("<Q",  width))
                f.write(struct.pack("<Q",  height))
                f.write(struct.pack("<4d", fx, fy, cx, cy))

    def _write_images_bin(self) -> None:
        with open(self.sparse_dir / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", len(self._images)))
            for (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name) in self._images:
                f.write(struct.pack("<I",  image_id))
                f.write(struct.pack("<4d", qw, qx, qy, qz))
                f.write(struct.pack("<3d", tx, ty, tz))
                f.write(struct.pack("<I",  camera_id))
                f.write(name.encode("utf-8") + b"\x00")
                f.write(struct.pack("<Q", 0))               # NUM_POINTS2D (none)

    def _write_points3d_bin(self) -> None:
        with open(self.sparse_dir / "points3D.bin", "wb") as f:
            f.write(struct.pack("<Q", 0))                   # empty, COLMAP fills this