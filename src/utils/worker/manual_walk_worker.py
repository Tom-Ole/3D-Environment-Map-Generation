import logging
import math
import time
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from bosdyn.client.frame_helpers import (
    get_a_tform_b,
    VISION_FRAME_NAME,
    BODY_FRAME_NAME,
)

from utils.route.manual_walk import ManualWalkInterface

logger = logging.getLogger(__name__)


def _xy_distance(pose_a, pose_b) -> float:
    """2D ground-plane distance between two SE3Poses (ignores Z/height)."""
    dx = pose_a.x - pose_b.x
    dy = pose_a.y - pose_b.y
    return math.sqrt(dx * dx + dy * dy)


class ManualWalkWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    image_captured = pyqtSignal(str)

    def __init__(self, interface: ManualWalkInterface, distance_interval_m: float = 1.0):
        super().__init__()
        self.interface = interface
        self.distance_interval_m = distance_interval_m

    def run(self):
        try:
            self.interface.start_walk(self.distance_interval_m)
            self.finished.emit()
        except Exception as e:
            logger.error("ManualWalkWorker error: %s", e)
            self.error.emit(e)

    def stop(self):
        self.interface.stop_walk()