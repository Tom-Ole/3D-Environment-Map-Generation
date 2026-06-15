"""CPU-based geometric reconstruction using SPOT poses + ORB triangulation.

No model download required.  Dependencies: opencv-python (already used by
the rest of the project).  Open3D is used for optional outlier removal.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ai_reconstruction.models.base import ReconstructionModel
from ai_reconstruction.types import AIPointCloudResult

logger = logging.getLogger(__name__)


class GeometricModel(ReconstructionModel):
    """
    Geometric Structure-from-Motion using known SPOT VIO poses.

    Because SPOT's onboard visual-inertial odometry already provides accurate
    metric poses, this model skips pose estimation entirely and focuses on
    triangulation:

    1. Load images and build 3x4 projection matrices [K @ R | t] from SPOT poses
    2. Detect ORB keypoints in each image
    3. Match keypoints between consecutive frame pairs (brute-force Hamming)
    4. Triangulate matched 2-D correspondences into 3-D points using known poses
    5. Filter by reprojection error and positive-depth constraint
    6. Colour each 3-D point from the source image

    The result is a sparse, metric-scale coloured point cloud that can be used
    directly or fed into a meshing step.

    Advantages over AI models:
    - Runs entirely on CPU — no GPU or model download needed
    - Output is metric-scale (inherits SPOT's metric poses)
    - Deterministic results
    - Very fast on small image sets

    Limitations:
    - Sparse reconstruction (feature-point density only)
    - Fails in textureless / dark areas where ORB finds no features
    - Quality degrades if SPOT poses are inaccurate (e.g. fast motion)
    """

    name = "geometric"
    description = "Geometric SfM with ORB triangulation (CPU, metric scale via SPOT poses)"

    def __init__(
        self,
        device: str = "cpu",
        spot_poses: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict] = None,
        image_timestamps: Optional[List[float]] = None,
        image_size: Optional[int] = None,
        min_matches: int = 20,
        max_reproj_error: float = 4.0,
        orb_features: int = 2000,
    ):
        super().__init__("cpu")  # always CPU
        self.spot_poses = spot_poses          # Nx8 [ts, x,y,z, qx,qy,qz,qw]
        self.intrinsics = intrinsics or {}
        self.image_timestamps = image_timestamps  # per-keyframe Unix timestamps
        self.image_size = image_size
        self.min_matches = min_matches
        self.max_reproj_error = max_reproj_error
        self.orb_features = orb_features

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        self._loaded = True

    def reconstruct(
        self,
        image_paths: List[Path],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> AIPointCloudResult:
        import cv2

        if not self._loaded:
            self.load()

        if len(image_paths) < 2:
            logger.warning("GeometricModel needs >= 2 images")
            return AIPointCloudResult(points=np.zeros((0, 3), np.float32), model_name=self.name)

        self._report(progress_cb, 5, f"Loading {len(image_paths)} images")
        images = self._load_images(image_paths)

        self._report(progress_cb, 15, "Building projection matrices from SPOT poses")
        proj_mats, world_rots, world_trans = self._build_projections(image_paths)

        orb = cv2.ORB_create(nfeatures=self.orb_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        all_pts: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        n_pairs = len(image_paths) - 1

        for i in range(n_pairs):
            pct = 20.0 + 75.0 * i / n_pairs
            self._report(progress_cb, pct, f"Triangulating pair {i+1}/{n_pairs}")

            P1, P2 = proj_mats[i], proj_mats[i + 1]
            if P1 is None or P2 is None:
                continue

            img1, img2 = images[i], images[i + 1]

            # Detect + match ORB keypoints
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                continue

            matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
            if len(matches) < self.min_matches:
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T   # 2xN
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T

            # Triangulate
            pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)   # 4xN
            pts3d = (pts4d[:3] / (pts4d[3:] + 1e-10)).T         # Nx3

            # Validate by reprojection error and positive depth
            keep = self._validate(pts3d, pts1.T, pts2.T, P1, P2)
            pts3d = pts3d[keep]
            if len(pts3d) == 0:
                continue

            # Colour from first image (convert to RGB)
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            cols = self._sample_colors(img1_rgb, pts1.T[keep])

            all_pts.append(pts3d.astype(np.float32))
            all_cols.append(cols)

        if not all_pts:
            logger.warning("No triangulated points — check poses and image overlap")
            return AIPointCloudResult(
                points=np.zeros((0, 3), np.float32),
                model_name=self.name,
                metric_scale=True,
            )

        points = np.concatenate(all_pts, axis=0)
        colors = np.concatenate(all_cols, axis=0)

        self._report(progress_cb, 97, "Removing outliers")
        points, colors = _statistical_filter(points, colors)

        self._report(progress_cb, 100, f"Done — {len(points):,} triangulated points")
        return AIPointCloudResult(
            points=points,
            colors=colors,
            image_paths=image_paths,
            model_name=self.name,
            metric_scale=True,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_images(self, paths: List[Path]) -> List[np.ndarray]:
        import cv2
        imgs = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                imgs.append(np.zeros((480, 640, 3), np.uint8))
                continue
            if self.image_size:
                h, w = img.shape[:2]
                scale = self.image_size / max(h, w)
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
            imgs.append(img)
        return imgs

    def _build_projections(self, paths: List[Path]):
        """
        Build 3x4 projection matrices P = K @ [R_cw | t_cw] for each frame.

        R_cw, t_cw are the world-to-camera rotation and translation derived
        by inverting each SPOT body pose.
        """
        proj_mats = []
        world_rots = []
        world_trans = []

        for i, path in enumerate(paths):
            K = self._get_K(path)
            R_cw, t_cw = self._get_world_to_camera(i)
            if R_cw is None:
                proj_mats.append(None)
                world_rots.append(None)
                world_trans.append(None)
                continue
            P = K @ np.hstack([R_cw, t_cw.reshape(3, 1)])
            proj_mats.append(P)
            world_rots.append(R_cw)
            world_trans.append(t_cw)

        return proj_mats, world_rots, world_trans

    def _get_K(self, path: Path) -> np.ndarray:
        """Return 3x3 camera matrix from stored intrinsics."""
        # Derive source name from filename: 00042_frontleft_fisheye_image.png
        parts = path.stem.split("_", 1)
        source = parts[1] if len(parts) == 2 else ""

        cal = self.intrinsics.get(source)
        if not cal:
            for k, v in self.intrinsics.items():
                if source in k or k in source:
                    cal = v
                    break

        if cal:
            return np.array([
                [float(cal.get("fx", 500)), 0., float(cal.get("cx", 320))],
                [0., float(cal.get("fy", 500)), float(cal.get("cy", 240))],
                [0., 0., 1.],
            ])
        # Default fallback for Spot fisheye (~500 px focal length)
        return np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])

    def _get_world_to_camera(self, frame_idx: int):
        """Return (R_cw, t_cw) world-to-camera transform for this keyframe."""
        from scipy.spatial.transform import Rotation

        if self.spot_poses is None or len(self.spot_poses) == 0:
            return np.eye(3), np.zeros(3)

        # Map frame index to the closest SPOT pose
        if self.image_timestamps and frame_idx < len(self.image_timestamps):
            target_ts = self.image_timestamps[frame_idx]
            pose_ts = self.spot_poses[:, 0]
            pose_idx = int(np.argmin(np.abs(pose_ts - target_ts)))
        else:
            # Uniform distribution across available poses
            n_poses = len(self.spot_poses)
            n_frames = max(len(self.image_timestamps or []), 1)
            pose_idx = min(int(frame_idx * n_poses / n_frames), n_poses - 1)

        pose = self.spot_poses[pose_idx]  # [ts, x, y, z, qx, qy, qz, qw]
        t_wb = pose[1:4]                  # body position in world frame
        q_wb = pose[4:8]                  # [x, y, z, w]

        # T_world <- body  →  invert to get T_camera <- world
        # Assuming camera frame = body frame (no extrinsic here)
        R_wb = Rotation.from_quat(q_wb).as_matrix()
        R_cw = R_wb.T
        t_cw = -R_cw @ t_wb
        return R_cw, t_cw

    def _validate(self, pts3d, pts1, pts2, P1, P2) -> np.ndarray:
        """Boolean mask: keep points with good reprojection and positive depth."""
        def proj(P, X):
            Xh = np.hstack([X, np.ones((len(X), 1))])
            p = (P @ Xh.T).T
            p[:, :2] /= p[:, 2:3]
            return p

        r1 = proj(P1, pts3d)
        r2 = proj(P2, pts3d)
        err1 = np.linalg.norm(r1[:, :2] - pts1, axis=1)
        err2 = np.linalg.norm(r2[:, :2] - pts2, axis=1)

        good_reproj = (err1 < self.max_reproj_error) & (err2 < self.max_reproj_error)
        pos_depth = (r1[:, 2] > 0) & (r2[:, 2] > 0)
        finite = np.isfinite(pts3d).all(axis=1)

        return good_reproj & pos_depth & finite

    def _sample_colors(self, img_rgb: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        px = np.clip(np.round(pts_2d[:, 0]).astype(int), 0, w - 1)
        py = np.clip(np.round(pts_2d[:, 1]).astype(int), 0, h - 1)
        return img_rgb[py, px]


def _statistical_filter(pts: np.ndarray, cols: np.ndarray):
    """Remove outlier points using Open3D statistical filter if available."""
    if len(pts) < 10:
        return pts, cols
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd_clean, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        idx_arr = np.asarray(idx)
        return np.asarray(pcd_clean.points, np.float32), cols[idx_arr]
    except Exception:
        return pts, cols
