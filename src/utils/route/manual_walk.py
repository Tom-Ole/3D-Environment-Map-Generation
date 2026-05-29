import logging
import math
import time
from typing import Optional

from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME

from utils.image.get_images import get_image
from utils.image.colmap_writer import ColmapWriter
from utils.image.ImageOptions import ImageOptions, ImageSources
from pathlib import Path

logger = logging.getLogger(__name__)


def _xy_distance(pose_a, pose_b) -> float:
    """2D ground-plane distance between two SE3Poses (ignores Z/height oscillation)."""
    dx = pose_a.x - pose_b.x
    dy = pose_a.y - pose_b.y
    return math.sqrt(dx * dx + dy * dy)


class ManualWalkInterface:
    def __init__(self, controller):
        self.controller = controller
        self._running = False
        self._distance_interval_m = 1.0
        self._last_pose = None


        self.frame_id = 0



    def start_walk(self, distance_m: float):
        """Begin the walk loop — call this from a worker thread."""
        self._distance_interval_m = distance_m
        self._last_pose = None
        self._running = True
        self.frame_id = 0

        logger.info("ManualWalkInterface started — capture every %.2f m", distance_m)

        while self._running:
            try:
                current_pose = self._get_body_pose()
            except Exception as e:
                logger.warning("Could not read robot pose: %s", e)
                time.sleep(0.1)
                continue

            if current_pose is None:
                time.sleep(0.1)
                continue

            if self._last_pose is None:
                self._last_pose = current_pose
                self.capture_image()
            elif _xy_distance(current_pose, self._last_pose) >= self._distance_interval_m:
                self._last_pose = current_pose
                self.capture_image()

            time.sleep(0.1)

        logger.info("ManualWalkInterface stopped")

    def stop_walk(self):
        """Signal the walk loop to exit on next iteration."""
        self._running = False

    def capture_image(self):
        """Trigger a single image capture at the current pose."""
        try:
            

            self.controller.get_image(self.frame_id)
            self.frame_id += 1
            logger.info(
                    "Image captured at (%.2f, %.2f) → %s",
                    self._last_pose.x,
                    self._last_pose.y,
                    self.controller.image_options.output_path
                )
        except Exception as e:
            logger.error("Image capture failed: %s", e)
            raise

    def _get_body_pose(self):
        """
        Return body pose in the vision frame (SE3Pose), or None if unavailable.
        Vision frame is preferred over odom — it fuses visual odometry and is
        more stable for distance-based triggering.
        """
        state = self.controller.robot_state_client.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        return get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)