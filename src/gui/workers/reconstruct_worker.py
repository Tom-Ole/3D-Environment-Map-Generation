"""Reconstruction worker thread for offline 3D reconstruction."""

import logging
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Slot

from reconstruction.pipeline import ReconstructionPipeline
from reconstruction.types import ReconstructionProgress

logger = logging.getLogger(__name__)


class ReconstructWorker(QThread):
    """Worker thread for offline 3D reconstruction pipeline."""

    progress = Signal(dict)  # Emits progress updates
    finished = Signal()  # Emits when reconstruction completes
    error = Signal(str)  # Emits error messages

    def __init__(
        self,
        session_path: Path,
        voxel_size: float = 0.05,
        loop_closure_threshold: float = 2.0,
        max_correspondence_distance: float = 0.1,
        icp_iterations: int = 50,
    ):
        """
        Initialize reconstruction worker.

        Args:
            session_path: Path to recorded session
            voxel_size: Voxel downsampling size (meters)
            loop_closure_threshold: Loop closure spatial threshold (meters)
            max_correspondence_distance: ICP correspondence distance (meters)
            icp_iterations: Number of ICP iterations per registration
        """
        super().__init__()
        self.session_path = Path(session_path)
        self.voxel_size = voxel_size
        self.loop_closure_threshold = loop_closure_threshold
        self.max_correspondence_distance = max_correspondence_distance
        self.icp_iterations = icp_iterations
        self.pipeline = None
        self.running = True

    def run(self) -> None:
        """Worker thread main loop."""
        try:
            logger.info(f"Starting reconstruction for {self.session_path}")

            # Validate session path
            if not self.session_path.exists():
                raise FileNotFoundError(f"Session path not found: {self.session_path}")

            # Create and run reconstruction pipeline
            self.pipeline = ReconstructionPipeline(
                session_path=self.session_path,
                voxel_size=self.voxel_size,
                loop_closure_threshold=self.loop_closure_threshold,
                max_correspondence_distance=self.max_correspondence_distance,
                icp_iterations=self.icp_iterations,
                progress_callback=self._on_progress,
            )

            success = self.pipeline.run()

            if success:
                logger.info("Reconstruction completed successfully")
                self.finished.emit()
            else:
                logger.error("Reconstruction pipeline failed")
                self.error.emit("Reconstruction pipeline failed - see logs for details")

        except FileNotFoundError as e:
            logger.error(f"Session not found: {e}")
            self.error.emit(f"Session not found: {self.session_path}")
        except Exception as e:
            logger.error(f"Reconstruction worker error: {e}", exc_info=True)
            self.error.emit(f"Reconstruction failed: {str(e)}")
        finally:
            self.running = False

    def _on_progress(self, progress: ReconstructionProgress) -> None:
        """
        Handle progress update from reconstruction pipeline.

        Args:
            progress: ReconstructionProgress object with step info
        """
        if not self.running:
            return

        progress_data = {
            "step_name": progress.step_name,
            "step_number": progress.step_number,
            "total_steps": progress.total_steps,
            "progress_pct": progress.progress_pct,
            "message": progress.message,
        }

        try:
            self.progress.emit(progress_data)
        except Exception as e:
            logger.warning(f"Failed to emit progress signal: {e}")

    @Slot()
    def stop(self) -> None:
        """Stop reconstruction gracefully."""
        logger.info("Stopping reconstruction")
        self.running = False
