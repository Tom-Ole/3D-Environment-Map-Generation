import logging
from typing import Optional

from PyQt5.QtCore import pyqtSignal

logger = logging.getLogger(__name__)


def report_error(error_signal: pyqtSignal, message: str, exc: Optional[BaseException] = None) -> None:
    if exc is not None:
        logger.error("%s", message, exc_info=exc)
    else:
        logger.error("%s", message)
    error_signal.emit(message)
