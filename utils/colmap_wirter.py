
import logging
from pathlib import Path
from typing import Dict
import numpy as np
from scipy.spatial.transform import Rotation


logger = logging.getLogger(__name__)

def matrix_to_colmap_pose(cam_to_world: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    """
    Convert a 4x4 camera-to-world matrix to the COLMAP extrinsic convention.
 
    COLMAP stores the *world-to-camera* transform as
        (qw, qx, qy, qz, tx, ty, tz)
    where  X_cam = R @ X_world + t.
    
    Returns
    ------- 
    qw, qx, qy, qz, tx, ty, tz  (all floats)
    """
    cam_to_world = np.linalg.inv(cam_to_world) # COLMAP uses world-to-camera convention
    R = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]

    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
    return float(qw), float(qx), float(qy), float(qz), float(t[0]), float(t[1]), float(t[2])

class ColmapWriter:

    def __init__(self, sparse_dir: Path) -> None:
        self.sparse_dir = sparse_dir
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        self._camera_ids: Dict[str, int] = {}
        self._image_id = 1

        self._init_files()


    def _init_files(self) -> None:
        """Create header-only files if they do not already exist"""

        files = {
            "camera.txt": (
                "# Camera list with one line of data per camera:\n"
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                "# PINHOLE model: fx fy cx cy\n"
                "#\n"
            ),
            "images.txt": (
                "# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                "#\n"
            ),
            "points3D.txt": (
                "# 3D point list – empty; populated by COLMAP reconstruction\n"
                "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
                "#\n"
            )
        }

        for filename, header in files.items():
            path = self.sparse_dir / filename
            if not path.exists():
                path.write_text(header)

    def register_camera(self, source_name: str, width: int, height: int, intrinsics: dict) -> int:
        """
        Register *source_name* as a COLMAP camera the first time it is seen.
        Subsequent calls with the same name just return the existing ID.
        """
        if source_name in self._camera_ids:
            return self._camera_ids[source_name]
        
        camera_id = len(self._camera_ids) + 1
        self._camera_ids[source_name] = camera_id

        fx = intrinsics.get("fx", 0.0)
        fy = intrinsics.get("fy", 0.0)
        cx = intrinsics.get("cx", width / 2.0)
        cy = intrinsics.get("cy", height / 2.0)
 
        line = (
            f"{camera_id} PINHOLE {width} {height} "
            f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
        )
        with open(self.sparse_dir / "cameras.txt", "a") as fh:
            fh.write(line)
 
        logger.debug("Registered camera %d: %s", camera_id, source_name)
        return camera_id
    
    def write_image(self, image_name: str, camera_id: int, cam_to_world: np.ndarray) -> int:
        """
        Append one image entry to images.txt.
 
        Parameters
        ----------
        image_name   : path relative to the ``images/`` directory,
                       e.g. ``"frontleft_fisheye_image/00001.jpg"``
        camera_id    : ID previously returned by :meth:`register_camera`
        cam_to_world : 4×4 camera-to-world transform
 
        Returns
        -------
        The COLMAP image ID assigned to this entry.

        """
        
        image_id = self._image_id
        self._image_id += 1

        qw, qx, qy, qz, tx, ty, tz = matrix_to_colmap_pose(cam_to_world)

        entry = (
            f"{image_id} "
            f"{qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{tx:.9f} {ty:.9f} {tz:.9f} "
            f"{camera_id} {image_name}\n"
            "\n"   # empty POINTS2D line - no 2D-3D correspondences yet
        )

        with open(self.sparse_dir / "images.txt", "a") as fh:
            fh.write(entry)
 
        return image_id

