"""DUSt3R reconstruction model wrapper.

Installation:
    pip install git+https://github.com/naver/dust3r.git

The model (~600 MB) is downloaded automatically from HuggingFace Hub on first
use.  A GPU with at least 6 GB VRAM is recommended; CPU inference is possible
but slow (minutes per image pair).
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ai_reconstruction.models.base import ReconstructionModel
from ai_reconstruction.types import AIPointCloudResult

logger = logging.getLogger(__name__)

_DEFAULT_CKPT = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"


class DUSt3RModel(ReconstructionModel):
    """
    Dense Unconstrained Stereo 3D Reconstruction (DUSt3R, CVPR 2024).

    DUSt3R feeds all image pairs through a ViT-L encoder / cross-attention
    decoder to predict per-pixel 3-D point maps.  A global alignment step
    (iterative bundle-adjustment on the point maps) then produces a single
    consistent scene reconstruction.

    Strengths:
    - Works on uncalibrated images (no known intrinsics required)
    - Dense reconstruction with per-pixel depth

    Limitations:
    - Output is up-to-scale (no metric depth without ground truth reference)
    - GPU memory scales with number of pairs (O(N^2) for complete graph)
    """

    name = "dust3r"
    description = "DUSt3R: Dense Unconstrained Stereo 3D Reconstruction (ViT-L, CVPR 2024)"

    def __init__(
        self,
        device: str = "auto",
        checkpoint: str = _DEFAULT_CKPT,
        image_size: int = 512,
        batch_size: int = 1,
        niter: int = 300,
        scene_graph: str = "complete",
    ):
        super().__init__(device)
        self.checkpoint = checkpoint
        self.image_size = image_size
        self.batch_size = batch_size
        self.niter = niter
        self.scene_graph = scene_graph
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import dust3r  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        from dust3r.model import AsymmetricCroCo3DStereo

        logger.info(f"Loading DUSt3R [{self.checkpoint}] on {self.device}")
        self._model = AsymmetricCroCo3DStereo.from_pretrained(self.checkpoint)
        self._model = self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info("DUSt3R model ready")

    def reconstruct(
        self,
        image_paths: List[Path],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> AIPointCloudResult:
        if not self._loaded:
            self.load()

        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        str_paths = [str(p) for p in image_paths]

        self._report(progress_cb, 5, f"Loading {len(image_paths)} images (size={self.image_size})")
        images = load_images(str_paths, size=self.image_size)

        self._report(progress_cb, 15, f"Building image pairs (graph={self.scene_graph})")
        pairs = make_pairs(images, scene_graph=self.scene_graph,
                           prefilter=None, symmetrize=True)

        self._report(progress_cb, 20, f"Running inference on {len(pairs)} pairs")
        output = inference(pairs, self._model, self.device,
                           batch_size=self.batch_size)

        self._report(progress_cb, 60, f"Global alignment ({self.niter} iters)")
        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
        )
        scene.compute_global_alignment(
            init="mst", niter=self.niter, schedule="cosine", lr=0.01
        )

        self._report(progress_cb, 90, "Extracting point cloud")

        pts3d = scene.get_pts3d()
        masks = scene.get_masks()
        imgs = scene.imgs
        poses = scene.get_im_poses()

        all_pts, all_cols = [], []
        for pts, mask, img in zip(pts3d, masks, imgs):
            pts_np = _to_numpy(pts)
            mask_np = _to_numpy(mask).astype(bool)
            img_np = _to_numpy(img)
            all_pts.append(pts_np[mask_np])
            all_cols.append((img_np[mask_np] * 255).clip(0, 255).astype(np.uint8))

        points = np.concatenate(all_pts, axis=0).astype(np.float32) if all_pts else np.zeros((0, 3), np.float32)
        colors = np.concatenate(all_cols, axis=0) if all_cols else None
        poses_np = _to_numpy(poses) if poses is not None else None

        self._report(progress_cb, 100, f"Done — {len(points):,} points")
        return AIPointCloudResult(
            points=points,
            colors=colors,
            camera_poses=poses_np,
            image_paths=image_paths,
            model_name=self.name,
            metric_scale=False,
        )


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
