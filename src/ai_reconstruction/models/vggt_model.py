"""VGGT reconstruction model wrapper.

Installation:
    pip install git+https://github.com/facebookresearch/vggt.git

VGGT requires PyTorch and a GPU with ~8 GB VRAM for the 1B parameter model.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ai_reconstruction.models.base import ReconstructionModel
from ai_reconstruction.types import AIPointCloudResult

logger = logging.getLogger(__name__)

_DEFAULT_VARIANT = "facebook/VGGT-1B"


class VGGTModel(ReconstructionModel):
    """
    Visual Geometry Grounded Transformer (VGGT, CVPR 2025).

    VGGT reconstructs a scene from a set of images in a single feed-forward
    pass — no iterative optimisation is required.  This makes it significantly
    faster than DUSt3R/MASt3R at the cost of some accuracy on large scenes.

    The model jointly predicts:
    - Per-image depth maps
    - Camera intrinsics and extrinsics (SE3 poses)
    - World-frame point clouds (via unprojection of predicted depths)

    Strengths:
    - Very fast (single forward pass, no iterative alignment)
    - Handles variable-length image sets efficiently

    Limitations:
    - Less accurate than MASt3R on complex scenes
    - Requires GPU for acceptable speed
    - Output scale may vary across image sets
    """

    name = "vggt"
    description = "VGGT: Visual Geometry Grounded Transformer (single-pass, CVPR 2025)"

    def __init__(
        self,
        device: str = "auto",
        model_variant: str = _DEFAULT_VARIANT,
        image_size: int = 518,
        chunk_size: int = 16,
    ):
        super().__init__(device)
        self.model_variant = model_variant
        self.image_size = image_size
        self.chunk_size = chunk_size
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import vggt  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        from vggt.models.vggt import VGGT

        logger.info(f"Loading VGGT [{self.model_variant}] on {self.device}")
        self._model = VGGT.from_pretrained(self.model_variant)
        self._model = self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info("VGGT model ready")

    def reconstruct(
        self,
        image_paths: List[Path],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> AIPointCloudResult:
        import torch

        if not self._loaded:
            self.load()

        self._report(progress_cb, 5, f"Preprocessing {len(image_paths)} images")
        images_torch = self._preprocess(image_paths).to(self.device)

        all_pts, all_cols, all_poses = [], [], []
        n_chunks = max(1, len(image_paths) // self.chunk_size + 1)

        for chunk_idx in range(0, len(image_paths), self.chunk_size):
            chunk_end = min(chunk_idx + self.chunk_size, len(image_paths))
            chunk_imgs = images_torch[chunk_idx:chunk_end]      # T x C x H x W
            chunk_paths = image_paths[chunk_idx:chunk_end]

            self._report(
                progress_cb,
                10 + 80 * chunk_end / len(image_paths),
                f"VGGT inference: frames {chunk_idx+1}–{chunk_end}/{len(image_paths)}",
            )

            with torch.no_grad():
                # VGGT expects batch dimension: 1 x T x C x H x W
                predictions = self._model(chunk_imgs.unsqueeze(0))

            depths = self._extract_depths(predictions, chunk_end - chunk_idx)
            poses = self._extract_poses(predictions, chunk_end - chunk_idx)

            for i in range(len(chunk_paths)):
                depth = depths[i] if i < len(depths) else None
                pose = poses[i] if (poses is not None and i < len(poses)) else None
                pts, cols = self._unproject(depth, chunk_imgs[i], pose)
                if len(pts) > 0:
                    all_pts.append(pts)
                    all_cols.append(cols)
                    if pose is not None:
                        all_poses.append(pose)

        points = np.concatenate(all_pts, axis=0).astype(np.float32) if all_pts else np.zeros((0, 3), np.float32)
        colors = np.concatenate(all_cols, axis=0) if all_cols else None
        poses_np = np.stack(all_poses, axis=0) if all_poses else None

        self._report(progress_cb, 100, f"Done — {len(points):,} points")
        return AIPointCloudResult(
            points=points,
            colors=colors,
            camera_poses=poses_np,
            image_paths=image_paths,
            model_name=self.name,
            metric_scale=False,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _preprocess(self, paths: List[Path]):
        import torch
        import cv2

        tensors = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((self.image_size, self.image_size, 3), np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            scale = self.image_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(img).float() / 255.0
            tensors.append(t.permute(2, 0, 1))  # 3 x H x W

        # Pad to same size for batching
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        padded = []
        for t in tensors:
            _, h, w = t.shape
            pad = torch.zeros(3, max_h, max_w)
            pad[:, :h, :w] = t
            padded.append(pad)

        return torch.stack(padded)   # T x 3 x H x W

    def _extract_depths(self, predictions, n: int) -> list:
        """Extract per-image depth maps from model output."""
        candidates = ["depth", "depth_map", "depths", "pred_depth"]
        for key in candidates:
            val = None
            if isinstance(predictions, dict):
                val = predictions.get(key)
            elif hasattr(predictions, key):
                val = getattr(predictions, key)
            if val is not None:
                val = val.squeeze(0)   # remove batch dim
                return [_to_numpy(val[i]) for i in range(min(n, val.shape[0]))]
        logger.warning("VGGT: could not find depth maps in output")
        return []

    def _extract_poses(self, predictions, n: int):
        """Extract camera-to-world poses from model output."""
        candidates = ["extrinsics", "camera_poses", "poses", "pred_extrinsics"]
        for key in candidates:
            val = None
            if isinstance(predictions, dict):
                val = predictions.get(key)
            elif hasattr(predictions, key):
                val = getattr(predictions, key)
            if val is not None:
                val = val.squeeze(0)
                return [_to_numpy(val[i]) for i in range(min(n, val.shape[0]))]
        return None

    def _unproject(self, depth, img_tensor, pose) -> tuple:
        """Unproject a depth map to a coloured 3-D point cloud."""
        if depth is None or depth.size == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

        depth = depth.squeeze()
        if depth.ndim != 2:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

        h, w = depth.shape
        # Approximate pinhole intrinsics from image dimensions
        fx = fy = max(h, w) * 0.8
        cx, cy = w / 2.0, h / 2.0

        yy, xx = np.mgrid[0:h, 0:w]
        z = depth.astype(np.float32)
        x = (xx - cx) / fx * z
        y = (yy - cy) / fy * z

        valid = np.isfinite(z) & (z > 0.05) & (z < 100.0)
        pts = np.stack([x[valid], y[valid], z[valid]], axis=1)

        # Extract image colors
        img_np = _to_numpy(img_tensor)
        if img_np.shape[0] in (1, 3):
            img_np = img_np.transpose(1, 2, 0)
        if img_np.shape[:2] != (h, w):
            import cv2
            img_np = cv2.resize(img_np, (w, h))
        cols = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)[valid]

        # Transform to world frame using camera-to-world pose
        if pose is not None and len(pts) > 0:
            pose_np = _to_numpy(pose) if not isinstance(pose, np.ndarray) else pose
            if pose_np.shape == (4, 4):
                ones = np.ones((len(pts), 1), np.float32)
                pts_h = np.hstack([pts, ones])
                pts = (pose_np.astype(np.float32) @ pts_h.T).T[:, :3]

        return pts.astype(np.float32), cols


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
