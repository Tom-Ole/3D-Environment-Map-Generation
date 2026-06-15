"""MASt3R reconstruction model wrapper.

Expects MASt3R cloned into `mast3r/` at the project root (with DUSt3R as a
submodule at `mast3r/dust3r/`) and its pip dependencies installed.
main.py adds both repos to sys.path at startup so `import mast3r` resolves
to the real Python package instead of the repo root namespace.

Checkpoint should be placed at:
    <project-root>/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ai_reconstruction.models.base import ReconstructionModel
from ai_reconstruction.types import AIPointCloudResult

logger = logging.getLogger(__name__)

# Resolve checkpoint relative to project root (src/ai_reconstruction/models/ → project/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CKPT = str(_PROJECT_ROOT / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")


class MASt3RModel(ReconstructionModel):
    """
    Mastering Monocular 3D Reconstruction (MASt3R, ECCV 2024).

    Produces metric-scale sparse point clouds via sparse global alignment.
    Requires the locally cloned repo at <project>/mast3r/ and a downloaded
    checkpoint .pth file.
    """

    name = "mast3r"
    description = "MASt3R: Mastering Monocular 3D Reconstruction (metric scale, ECCV 2024)"

    def __init__(
        self,
        device: str = "auto",
        checkpoint: str = _DEFAULT_CKPT,
        image_size: int = 512,
        batch_size: int = 1,
        scene_graph: str = "complete",
        min_conf_thr: float = 1.5,
    ):
        super().__init__(device)
        self.checkpoint = checkpoint
        self.image_size = image_size
        self.batch_size = batch_size
        self.scene_graph = scene_graph
        self.min_conf_thr = min_conf_thr
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from mast3r.model import AsymmetricMASt3R  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        from mast3r.model import AsymmetricMASt3R

        ckpt = Path(self.checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"MASt3R checkpoint not found: {ckpt}\n"
                "Download it from https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
                "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
                f" and place it at {_DEFAULT_CKPT}"
            )

        logger.info(f"Loading MASt3R from [{ckpt}] on {self.device}")
        self._model = AsymmetricMASt3R.from_pretrained(str(ckpt))
        self._model = self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info("MASt3R model ready")

    def reconstruct(
        self,
        image_paths: List[Path],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> AIPointCloudResult:
        if not self._loaded:
            self.load()

        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

        str_paths = [str(p) for p in image_paths]

        self._report(progress_cb, 5, f"Loading {len(image_paths)} images (size={self.image_size})")
        images = load_images(str_paths, size=self.image_size)

        self._report(progress_cb, 15, f"Building image pairs (graph={self.scene_graph})")
        pairs = make_pairs(images, scene_graph=self.scene_graph,
                           prefilter=None, symmetrize=True)

        self._report(progress_cb, 20, f"Running MASt3R inference on {len(pairs)} pairs")
        output = inference(pairs, self._model, self.device,
                           batch_size=self.batch_size)

        self._report(progress_cb, 60, "Sparse global alignment (metric scale)")
        # sparse_global_alignment writes intermediate tensors to cache_path —
        # it cannot accept None; use a temp dir and clean up afterward.
        cache_dir = tempfile.mkdtemp(prefix="mast3r_cache_")
        try:
            scene = sparse_global_alignment(
                str_paths,
                output,
                cache_dir,
                model=self._model,
                device=self.device,
                min_conf_thr=self.min_conf_thr,
            )
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

        self._report(progress_cb, 90, "Extracting point cloud")

        # SparseGA stores sparse anchor points and matching colours separately
        pts3d = scene.get_sparse_pts3d()        # list of N_i×3 tensors
        pts_colors = scene.get_pts3d_colors()   # list of N_i×3 float32 arrays (0–1)
        poses = scene.get_im_poses()            # M×4×4 cam-to-world tensor

        all_pts, all_cols = [], []
        for pts, col in zip(pts3d, pts_colors):
            all_pts.append(_to_numpy(pts))
            all_cols.append((np.asarray(col) * 255).clip(0, 255).astype(np.uint8))

        points = np.concatenate(all_pts, axis=0).astype(np.float32) if all_pts else np.zeros((0, 3), np.float32)
        colors = np.concatenate(all_cols, axis=0) if all_cols else None
        poses_np = _to_numpy(poses) if poses is not None else None

        self._report(progress_cb, 100, f"Done — {len(points):,} points (metric)")
        return AIPointCloudResult(
            points=points,
            colors=colors,
            camera_poses=poses_np,
            image_paths=image_paths,
            model_name=self.name,
            metric_scale=True,
        )


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
