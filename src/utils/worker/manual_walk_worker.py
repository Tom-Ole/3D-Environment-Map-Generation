import logging
import time
from typing import Callable, Optional

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class ManualWalkWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    image_captured = pyqtSignal(str)

    def __init__(self, controller, distance_interval_m: float = 1.0):
        super().__init__()
        self.controller = controller
        self.distance_interval_m = distance_interval_m
        self._running = False
        self._last_position = None

    def run(self):
        try:
            self._running = True
            logger.info("Starting manual walk with distance interval: %.2fm", self.distance_interval_m)
            
            while self._running:
                if not self.controller.robot:
                    time.sleep(0.1)
                    continue
                
                try:
                    state = self.controller.robot_state_client.get_robot_state()
                    current_position = state.kinematic_state.odom_tform_body.position
                    
                    if self._last_position is not None:
                        distance = self._calculate_distance(self._last_position, current_position)
                        if distance >= self.distance_interval_m:
                            self._capture_image()
                            self._last_position = current_position
                    else:
                        self._last_position = current_position
                        self._capture_image()
                    
                except Exception as e:
                    logger.warning("Failed to get robot state during manual walk: %s", e)
                
                time.sleep(0.1)
            
            logger.info("Manual walk stopped")
            self.finished.emit()
            
        except Exception as e:
            logger.error("Manual walk worker error: %s", e)
            self.error.emit(e)

    def _calculate_distance(self, pos1, pos2):
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        return (dx**2 + dy**2 + dz**2)**0.5

    def _capture_image(self):
        try:
            image_path = self.controller.get_image(save=True)
            if image_path:
                self.image_captured.emit(str(image_path))
                logger.info("Image captured during manual walk")
        except Exception as e:
            logger.error("Failed to capture image during manual walk: %s", e)

    def stop(self):
        self._running = False
        self.wait()
