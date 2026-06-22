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
        intrinsics: Optional[dict] = None,
    ):
        super().__init__(device)
        self.checkpoint = checkpoint
        self.image_size = image_size
        self.batch_size = batch_size
        self.scene_graph = scene_graph
        self.min_conf_thr = min_conf_thr
        # source_name -> {fx, fy, cx, cy, distortion}; used to rectify fisheye
        # images to a pinhole model before reconstruction. Injected by the
        # pipeline (see _construct in models/__init__.py).
        self.intrinsics = intrinsics or {}
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

        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

        cache_dir = None
        undist_dir = None
        try:
            # Rectify SPOT fisheye -> pinhole. MASt3R assumes a near-pinhole
            # camera and fits a single focal; raw fisheye distortion warps the
            # reconstructed geometry. No-op for already-pinhole inputs.
            str_paths, undist_dir = self._prepare_images(image_paths, progress_cb)

            self._report(progress_cb, 8,
                         f"Loading {len(str_paths)} images (size={self.image_size})")
            images = load_images(str_paths, size=self.image_size)

            self._report(progress_cb, 15, f"Building image pairs (graph={self.scene_graph})")
            pairs = make_pairs(images, scene_graph=self.scene_graph,
                               prefilter=None, symmetrize=True)

            self._report(progress_cb, 20,
                         f"Sparse global alignment on {len(pairs)} pairs (metric scale)")
            # MASt3R runs the forward pass internally inside
            # sparse_global_alignment (via forward_mast3r) -- it takes the raw
            # `pairs` from make_pairs, NOT a separate dust3r inference() output.
            # It writes intermediate tensors to cache_path (cannot be None).
            cache_dir = tempfile.mkdtemp(prefix="mast3r_cache_")
            scene = sparse_global_alignment(
                str_paths,
                pairs,
                cache_dir,
                model=self._model,
                device=self.device,
            )

            self._report(progress_cb, 90, "Extracting dense point cloud")
            points, colors, confidence = self._extract_dense(scene)
            poses = scene.get_im_poses()            # M x 4 x 4 cam-to-world
            poses_np = _to_numpy(poses) if poses is not None else None
        finally:
            if cache_dir:
                shutil.rmtree(cache_dir, ignore_errors=True)
            if undist_dir:
                shutil.rmtree(undist_dir, ignore_errors=True)

        self._report(progress_cb, 100, f"Done -- {len(points):,} dense points (metric)")
        return AIPointCloudResult(
            points=points,
            colors=colors,
            confidence=confidence,
            camera_poses=poses_np,
            image_paths=image_paths,
            model_name=self.name,
            metric_scale=True,
        )

    # -- helpers ----------------------------------------------------------------

    def _prepare_images(self, image_paths, progress_cb):
        """Undistort fisheye inputs to a temp dir; return (str_paths, temp_dir).

        temp_dir is None when no rectification was performed (caller skips
        cleanup). Falls back to the raw images if undistortion errors out.
        """
        if not self.intrinsics:
            return [str(p) for p in image_paths], None

        from ai_reconstruction.undistort import has_fisheye, undistort_images

        sources = [_source_from_path(p) for p in image_paths]
        if not has_fisheye(sources, self.intrinsics):
            return [str(p) for p in image_paths], None

        self._report(progress_cb, 4, "Undistorting fisheye images -> pinhole")
        out_dir = tempfile.mkdtemp(prefix="mast3r_undist_")
        try:
            rect = undistort_images(image_paths, sources, self.intrinsics, Path(out_dir))
            return [str(p) for p in rect], out_dir
        except Exception as e:
            logger.warning(f"Fisheye undistortion failed ({e}); using raw images")
            shutil.rmtree(out_dir, ignore_errors=True)
            return [str(p) for p in image_paths], None

    def _extract_dense(self, scene):
        """Densify per-view depthmaps into a world-frame coloured point cloud.

        get_sparse_pts3d() would return only the thin set of anchor
        correspondences used during optimisation -- not the scene geometry.
        Mirrors MASt3R's own demo (get_3D_model_from_scene): keep pixels whose
        cleaned confidence exceeds the threshold, colour them from scene.imgs.
        """
        dense_pts, _, dense_confs = scene.get_dense_pts3d(clean_depth=True)
        rgb_imgs = scene.imgs                    # list of H x W x 3 float [0,1]

        all_pts, all_cols, all_conf = [], [], []
        for pts_i, conf_i, img_i in zip(dense_pts, dense_confs, rgb_imgs):
            conf_i = _to_numpy(conf_i)           # H x W
            mask = conf_i > self.min_conf_thr    # drops low-conf / invalid depth
            if not mask.any():
                continue
            pts_i = _to_numpy(pts_i).reshape(-1, 3)[mask.ravel()]
            col_i = (np.asarray(img_i)[mask] * 255).clip(0, 255).astype(np.uint8)
            all_pts.append(pts_i.astype(np.float32))
            all_cols.append(col_i)
            all_conf.append(conf_i[mask].astype(np.float32))

        if not all_pts:
            return np.zeros((0, 3), np.float32), None, None
        points = np.concatenate(all_pts, axis=0)
        colors = np.concatenate(all_cols, axis=0)
        confidence = np.concatenate(all_conf, axis=0)
        return points, colors, confidence


def _source_from_path(path: Path) -> str:
    """Parse the camera source from a '{frame:05d}_{source}.png' filename."""
    parts = path.stem.split("_", 1)
    return parts[1] if len(parts) == 2 else ""


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
