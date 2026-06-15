"""AI reconstruction pipeline orchestrator.

Six-stage pipeline:
  1. load_session    — images, intrinsics, optional SPOT poses
  2. select_keyframes — motion- or interval-based subset
  3. load_model      — download / initialise model weights
  4. inference       — model reconstruction → raw AIPointCloudResult
  5. postprocess     — confidence filter + depth clamp + voxel downsample
  6. export          — save PLY cloud, camera poses, metadata.json
"""

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ai_reconstruction.export import export_results
from ai_reconstruction.image_loader import (
    list_session_images,
    load_camera_intrinsics,
)
from ai_reconstruction.keyframe import select_keyframes
from ai_reconstruction.models import get_model
from ai_reconstruction.types import (
    AIPointCloudResult,
    AIReconstructionConfig,
    AIReconstructionProgress,
    AIReconstructionResult,
    ImageRecord,
)
from recording.session import load_poses

logger = logging.getLogger(__name__)

_N_STAGES = 6


class AIReconstructionPipeline:
    """
    Offline AI-based 3D reconstruction pipeline for SPOT camera data.

    Usage:
        config = AIReconstructionConfig(model_type=ModelType.AUTO, ...)
        pipeline = AIReconstructionPipeline(session_path, config, cb)
        result = pipeline.run()
    """

    def __init__(
        self,
        session_path: Path,
        config: Optional[AIReconstructionConfig] = None,
        progress_callback: Optional[Callable[[AIReconstructionProgress], None]] = None,
    ):
        self.session_path = Path(session_path)
        self.config = config or AIReconstructionConfig()
        self.progress_callback = progress_callback

        # Populated during stages
        self._image_records: List[ImageRecord] = []
        self._keyframes: List[ImageRecord] = []
        self._spot_poses: Optional[np.ndarray] = None
        self._intrinsics: dict = {}
        self._model = None
        self._start_time: float = 0.0

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> AIReconstructionResult:
        """
        Execute all pipeline stages in sequence.

        Returns:
            AIReconstructionResult with success flag and output paths.
        """
        self._start_time = time.time()

        # Stage 1
        self._emit("load_session", 1, 0.0, "Loading session data…")
        if not self._load_session():
            return AIReconstructionResult(error_message="Stage 'load_session' failed")

        # Stage 2
        self._emit("select_keyframes", 2, 0.0, "Selecting keyframes…")
        if not self._select_keyframes():
            return AIReconstructionResult(error_message="Stage 'select_keyframes' failed")

        # Stage 3
        self._emit("load_model", 3, 0.0, "Loading AI model…")
        if not self._load_model():
            return AIReconstructionResult(error_message="Stage 'load_model' failed")

        # Stage 4
        self._emit("inference", 4, 0.0, "Running model inference…")
        raw = self._run_inference()
        if raw is None:
            return AIReconstructionResult(error_message="Stage 'inference' failed")

        # Stage 5
        self._emit("postprocess", 5, 0.0, "Post-processing point cloud…")
        raw = self._postprocess(raw)

        # Stage 6
        self._emit("export", 6, 0.0, "Exporting results…")
        return self._export(raw)

    # ── Stage implementations ─────────────────────────────────────────────────

    def _load_session(self) -> bool:
        """Discover images, load intrinsics and SPOT poses from session."""
        sources = self.config.camera_sources or None
        self._image_records = list_session_images(self.session_path, sources=sources)

        if not self._image_records:
            logger.error(
                f"No images found in {self.session_path}/images/. "
                f"Sources requested: {sources}"
            )
            return False

        self._intrinsics = load_camera_intrinsics(self.session_path)

        try:
            self._spot_poses = load_poses(self.session_path)
        except Exception as e:
            logger.warning(f"Could not load SPOT poses: {e}")
            self._spot_poses = None

        self._emit(
            "load_session", 1, 100.0,
            f"Loaded {len(self._image_records)} images, "
            f"{'poses available' if self._spot_poses is not None else 'no SPOT poses'}"
        )
        return True

    def _select_keyframes(self) -> bool:
        """Apply the configured keyframe selection strategy."""
        self._keyframes = select_keyframes(
            records=self._image_records,
            strategy=self.config.keyframe_strategy,
            max_frames=self.config.max_images,
            interval=self.config.keyframe_interval,
            spot_poses=self._spot_poses,
            min_translation=self.config.keyframe_min_translation,
            min_rotation_deg=self.config.keyframe_min_rotation_deg,
        )
        if not self._keyframes:
            logger.error("Keyframe selection returned no frames")
            return False

        self._emit(
            "select_keyframes", 2, 100.0,
            f"Selected {len(self._keyframes)} keyframes from "
            f"{len(self._image_records)} images"
        )
        return True

    def _load_model(self) -> bool:
        """Instantiate and load the selected AI/geometric model."""
        model_type = self.config.model_type.value
        device = self.config.device.value

        # Build kwargs; _construct() in the registry discards what the model
        # doesn't accept, so we can safely pass everything.
        timestamps = [r.inferred_timestamp for r in self._keyframes]
        extra = dict(
            spot_poses=self._spot_poses,
            intrinsics=self._intrinsics,
            image_timestamps=timestamps,
            image_size=self.config.image_size,
        )

        self._emit("load_model", 3, 10.0, f"Instantiating model: {model_type}")
        self._model = get_model(model_type, device=device, **extra)

        if self._model is None:
            logger.error(f"Could not instantiate model '{model_type}'")
            return False

        self._emit("load_model", 3, 30.0, f"Loading weights for {self._model.name}…")
        self._model.load()

        self._emit(
            "load_model", 3, 100.0,
            f"Model ready: {self._model.name} on {self._model.device}"
        )
        return True

    def _run_inference(self) -> Optional[AIPointCloudResult]:
        """Call model.reconstruct() with a progress callback wired to this stage."""
        image_paths = [r.path for r in self._keyframes]

        def _cb(pct: float, msg: str) -> None:
            self._emit("inference", 4, pct, msg)

        try:
            result = self._model.reconstruct(image_paths, progress_cb=_cb)
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return None

        n_pts = len(result.points) if result.points is not None else 0
        self._emit("inference", 4, 100.0, f"Inference complete — {n_pts:,} raw points")
        return result

    def _postprocess(self, raw: AIPointCloudResult) -> AIPointCloudResult:
        """Filter and downsample the raw point cloud."""
        if raw.points is None or len(raw.points) == 0:
            logger.warning("Post-process: empty point cloud — skipping")
            return raw

        pts = raw.points.copy()
        cols = raw.colors.copy() if raw.colors is not None else None
        conf = raw.confidence.copy() if raw.confidence is not None else None

        self._emit("postprocess", 5, 10.0, f"Input: {len(pts):,} points")

        # 1. Confidence filtering (DUSt3R / MASt3R)
        if conf is not None and self.config.confidence_threshold > 0:
            mask = conf >= self.config.confidence_threshold
            pts, cols, conf = _mask(pts, cols, conf, mask)
            self._emit("postprocess", 5, 30.0,
                       f"After confidence filter: {len(pts):,} points")

        # 2. Depth range clamp (skip if most z-values are near 0, e.g. metric XYZ)
        z = pts[:, 2]
        depth_range_ratio = np.mean(
            (z > self.config.min_depth) & (z < self.config.max_depth)
        )
        if depth_range_ratio > 0.5:
            mask = (z > self.config.min_depth) & (z < self.config.max_depth)
            pts, cols, conf = _mask(pts, cols, conf, mask)
            self._emit("postprocess", 5, 50.0,
                       f"After depth clamp: {len(pts):,} points")

        # 3. Voxel downsample
        if self.config.voxel_size > 0 and len(pts) > 5000:
            pts, cols = _voxel_downsample(pts, cols, self.config.voxel_size)
            self._emit("postprocess", 5, 75.0,
                       f"After voxel downsample ({self.config.voxel_size} m): {len(pts):,}")

        # 4. Statistical outlier removal
        if len(pts) > 50:
            pts, cols = _remove_outliers(pts, cols)

        raw.points = pts
        raw.colors = cols
        raw.confidence = conf

        self._emit("postprocess", 5, 100.0, f"Post-processed: {len(pts):,} points")
        return raw

    def _export(self, raw: AIPointCloudResult) -> AIReconstructionResult:
        """Persist results and return summary."""
        duration = time.time() - self._start_time
        try:
            final = export_results(
                result=raw,
                config=self.config,
                session_path=self.session_path,
                duration=duration,
                keyframe_count=len(self._keyframes),
            )
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return AIReconstructionResult(error_message=str(e))

        self._emit("export", 6, 100.0,
                   f"Saved to {self.session_path / 'ai_reconstruction'}")
        return final

    # ── Progress helper ───────────────────────────────────────────────────────

    def _emit(self, stage: str, idx: int, pct: float, msg: str) -> None:
        overall = ((idx - 1) + pct / 100.0) / _N_STAGES * 100.0
        prog = AIReconstructionProgress(
            stage=stage,
            stage_index=idx,
            total_stages=_N_STAGES,
            stage_pct=pct,
            overall_pct=overall,
            message=msg,
        )
        logger.info(f"[AI-{idx}/{_N_STAGES}] {stage} {pct:.0f}%  {msg}")
        if self.progress_callback:
            try:
                self.progress_callback(prog)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")


# ── Utility functions ─────────────────────────────────────────────────────────

def _mask(pts, cols, conf, mask):
    pts = pts[mask]
    cols = cols[mask] if cols is not None else None
    conf = conf[mask] if conf is not None else None
    return pts, cols, conf


def _voxel_downsample(pts: np.ndarray, cols, voxel: float):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
        ds = pcd.voxel_down_sample(voxel)
        pts_out = np.asarray(ds.points, np.float32)
        cols_out = (np.asarray(ds.colors) * 255).astype(np.uint8) if cols is not None else None
        return pts_out, cols_out
    except Exception:
        return pts, cols


def _remove_outliers(pts: np.ndarray, cols):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd_c, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        idx_arr = np.asarray(idx)
        pts_out = np.asarray(pcd_c.points, np.float32)
        cols_out = cols[idx_arr] if cols is not None else None
        return pts_out, cols_out
    except Exception:
        return pts, cols
