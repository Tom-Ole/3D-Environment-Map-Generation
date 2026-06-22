"""QThread wrapper for the mesh colorization pipeline."""

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class ColorizeWorker(QThread):
    """
    Background thread that runs colorize_mesh() and emits progress signals.

    Signals
    -------
    finished(str)   Emitted with the output PLY path on success.
    error(str)      Emitted with an error message on failure.
    log(str)        Emitted for each log-level progress message.
    """

    finished = Signal(str)   # output PLY path
    error = Signal(str)
    log = Signal(str)

    def __init__(
        self,
        session_path: Path,
        cameras: Optional[List[str]] = None,
        max_images_per_camera: Optional[int] = None,
    ):
        super().__init__()
        self.session_path = Path(session_path)
        self.cameras = cameras
        self.max_images_per_camera = max_images_per_camera

    def run(self) -> None:
        try:
            self.log.emit(f"Colorizing mesh for session: {self.session_path.name}")

            from reconstruction.colorize import colorize_mesh

            output_path = colorize_mesh(
                session_path=self.session_path,
                cameras=self.cameras,
                max_images_per_camera=self.max_images_per_camera,
            )

            self.log.emit(f"Coloured mesh saved → {output_path}")
            self.finished.emit(str(output_path))

        except FileNotFoundError as exc:
            msg = str(exc)
            logger.error(msg)
            self.error.emit(msg)
        except Exception as exc:
            msg = f"Colorization failed: {exc}"
            logger.error(msg, exc_info=True)
            self.error.emit(msg)
