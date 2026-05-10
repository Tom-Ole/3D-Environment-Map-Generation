import struct
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# COLMAP PINHOLE model (fx fy cx cy) — model_id = 1
_PINHOLE_MODEL_ID  = 1
_PINHOLE_NUM_PARAMS = 4


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
    Buffers camera and image data in memory, then writes the COLMAP binary
    sparse model (cameras.bin / images.bin / points3D.bin) on save().

    Binary layout (all little-endian):

    cameras.bin
        uint64  num_cameras
        per camera:
            uint32  camera_id
            int32   model_id        (1 = PINHOLE)
            uint64  width
            uint64  height
            double[4] params        (fx, fy, cx, cy)

    images.bin
        uint64  num_images
        per image:
            uint32  image_id
            double[4] qvec          (qw, qx, qy, qz)
            double[3] tvec          (tx, ty, tz)
            uint32  camera_id
            char[]  name            (null-terminated UTF-8)
            uint64  num_points2d    (0 — no keypoints at capture time)

    points3D.bin
        uint64  num_points3d        (0 — filled by COLMAP reconstruction)
    """

    def __init__(self, sparse_dir: Path) -> None:
        self.sparse_dir = Path(sparse_dir)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        # source_name → (camera_id, width, height, intrinsics_dict)
        self._cameras: Dict[str, Tuple[int, int, int, dict]] = {}
        self._next_camera_id = 1

        # ordered list of image tuples
        self._images: List[Tuple] = []
        self._next_image_id = 1

    # ── public API ────────────────────────────────────────────────────────────

    def register_camera(
        self, source_name: str, width: int, height: int, intrinsics: dict
    ) -> int:
        """
        Register a camera the first time it is seen; idempotent on repeats.
        Returns the assigned COLMAP camera_id.
        """
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
        image_name is relative to the images/ directory,
        e.g. 'frontleft_fisheye_image/00042.jpg'.
        Returns the assigned COLMAP image_id.
        """
        image_id = self._next_image_id
        self._next_image_id += 1
        pose = matrix_to_colmap_pose(cam_to_world)
        self._images.append((image_id, *pose, camera_id, image_name))
        logger.debug("Buffered image %d: %s", image_id, image_name)
        return image_id

    def save(self) -> None:
        """Flush all buffered data to binary files in sparse_dir."""
        self._write_cameras_bin()
        self._write_images_bin()
        self._write_points3d_bin()
        logger.info(
            "COLMAP model saved → %s  (%d camera(s), %d image(s))",
            self.sparse_dir, len(self._cameras), len(self._images),
        )

    # ── context-manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.save()

    # ── binary writers ────────────────────────────────────────────────────────

    def _write_cameras_bin(self) -> None:
        with open(self.sparse_dir / "cameras.bin", "wb") as f:
            f.write(struct.pack("<Q", len(self._cameras)))
            for _, (camera_id, width, height, intr) in self._cameras.items():
                fx = intr.get("fx", 0.0)
                fy = intr.get("fy", 0.0)
                cx = intr.get("cx", width  / 2.0)
                cy = intr.get("cy", height / 2.0)

                f.write(struct.pack("<I",  camera_id))           # CAMERA_ID  uint32
                f.write(struct.pack("<i",  _PINHOLE_MODEL_ID))   # MODEL_ID   int32
                f.write(struct.pack("<Q",  width))               # WIDTH      uint64
                f.write(struct.pack("<Q",  height))              # HEIGHT     uint64
                f.write(struct.pack("<4d", fx, fy, cx, cy))     # PARAMS     double[4]

    def _write_images_bin(self) -> None:
        with open(self.sparse_dir / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", len(self._images)))
            for (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name) in self._images:
                f.write(struct.pack("<I",  image_id))            # IMAGE_ID   uint32
                f.write(struct.pack("<4d", qw, qx, qy, qz))     # QVEC       double[4]
                f.write(struct.pack("<3d", tx, ty, tz))          # TVEC       double[3]
                f.write(struct.pack("<I",  camera_id))           # CAMERA_ID  uint32
                f.write(name.encode("utf-8") + b"\x00")          # NAME       null-term
                f.write(struct.pack("<Q", 0))                    # NUM_PTS2D  uint64 (none)

    def _write_points3d_bin(self) -> None:
        with open(self.sparse_dir / "points3D.bin", "wb") as f:
            f.write(struct.pack("<Q", 0))                        # NUM_PTS3D  uint64 (empty)