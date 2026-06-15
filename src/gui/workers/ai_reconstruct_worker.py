"""Worker thread for the AI reconstruction pipeline."""

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal, Slot

from ai_reconstruction.pipeline import AIReconstructionPipeline
from ai_reconstruction.types import (
    AIReconstructionConfig,
    AIReconstructionProgress,
    DeviceType,
    KeyframeStrategy,
    ModelType,
)

logger = logging.getLogger(__name__)


class AIReconstructWorker(QThread):
    """
    Runs AIReconstructionPipeline in a background thread.

    Signals:
        progress(dict)  — emitted after each stage step
        finished(dict)  — emitted on success, carries result statistics
        error(str)      — emitted on unrecoverable failure
    """

    progress = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        session_path: Path,
        model_type: str = "auto",
        device: str = "auto",
        image_size: int = 512,
        keyframe_strategy: str = "interval",
        keyframe_interval: int = 5,
        keyframe_min_translation: float = 0.3,
        keyframe_min_rotation_deg: float = 10.0,
        max_images: int = 100,
        camera_sources: Optional[List[str]] = None,
        voxel_size: float = 0.05,
        confidence_threshold: float = 1.5,
        global_alignment_iter: int = 300,
    ):
        super().__init__()
        self.session_path = Path(session_path)
        self._running = True

        self._config = AIReconstructionConfig(
            model_type=ModelType(model_type),
            device=DeviceType(device),
            image_size=image_size,
            keyframe_strategy=KeyframeStrategy(keyframe_strategy),
            keyframe_interval=keyframe_interval,
            keyframe_min_translation=keyframe_min_translation,
            keyframe_min_rotation_deg=keyframe_min_rotation_deg,
            max_images=max_images,
            camera_sources=camera_sources or [
                "frontleft_fisheye_image",
                "frontright_fisheye_image",
            ],
            voxel_size=voxel_size,
            confidence_threshold=confidence_threshold,
            global_alignment_iter=global_alignment_iter,
        )

    def run(self) -> None:
        try:
            if not self.session_path.exists():
                raise FileNotFoundError(f"Session not found: {self.session_path}")

            pipeline = AIReconstructionPipeline(
                session_path=self.session_path,
                config=self._config,
                progress_callback=self._on_progress,
            )

            result = pipeline.run()

            if not self._running:
                return

            if result.success:
                logger.info(
                    f"AI reconstruction complete: {result.point_count:,} pts, "
                    f"{result.duration_seconds:.1f}s"
                )
                self.finished.emit({
                    "point_count": result.point_count,
                    "keyframe_count": result.keyframe_count,
                    "model_used": result.model_used,
                    "device_used": result.device_used,
                    "duration_seconds": result.duration_seconds,
                    "point_cloud_path": str(result.point_cloud_path)
                    if result.point_cloud_path else "",
                })
            else:
                self.error.emit(result.error_message or "AI reconstruction failed")

        except FileNotFoundError as e:
            logger.error(str(e))
            self.error.emit(str(e))
        except Exception as e:
            logger.error(f"AIReconstructWorker: {e}", exc_info=True)
            self.error.emit(f"AI reconstruction failed: {e}")
        finally:
            self._running = False

    def _on_progress(self, prog: AIReconstructionProgress) -> None:
        if not self._running:
            return
        try:
            self.progress.emit({
                "stage": prog.stage,
                "stage_index": prog.stage_index,
                "total_stages": prog.total_stages,
                "stage_pct": prog.stage_pct,
                "overall_pct": prog.overall_pct,
                "message": prog.message,
            })
        except Exception as e:
            logger.warning(f"Progress emit failed: {e}")

    @Slot()
    def stop(self) -> None:
        logger.info("AIReconstructWorker stop requested")
        self._running = False
