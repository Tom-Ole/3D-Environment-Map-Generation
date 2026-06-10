"""Capture worker thread for real-time SPOT sensor streaming."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal, Slot

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient

from capture.lidar_client import LidarClientWrapper
from capture.image_client import ImageClientWrapper
from capture.state_client import StateClientWrapper
from capture.types import LidarFrame, CameraFrame, RobotPose
from recording.session import create_session, save_session_metadata, save_intrinsics
from recording.writer import DiskWriter

logger = logging.getLogger(__name__)


class CaptureWorker(QThread):
    """Worker thread for live LiDAR + camera + pose capture from SPOT."""

    connected = Signal()
    disconnected = Signal()
    error = Signal(str)

    def __init__(
        self,
        hostname: str,
        username: str,
        password: str,
        output_dir: Path,
        lidar_rate_hz: float = 10.0,
        camera_rate_hz: float = 5.0,
    ):
        """
        Initialize capture worker.

        Args:
            hostname: Robot hostname/IP
            username: Robot username
            password: Robot password
            output_dir: Output directory for sessions
            lidar_rate_hz: Desired LiDAR polling rate
            camera_rate_hz: Desired camera polling rate
        """
        super().__init__()
        self.hostname = hostname
        self.username = username
        self.password = password
        self.output_dir = Path(output_dir)
        self.lidar_rate_hz = lidar_rate_hz
        self.camera_rate_hz = camera_rate_hz

        # Robot and clients
        self.sdk = None
        self.robot = None
        self.lidar_client_wrapper: Optional[LidarClientWrapper] = None
        self.image_client_wrapper: Optional[ImageClientWrapper] = None
        self.state_client_wrapper: Optional[StateClientWrapper] = None

        # State
        self.running = False
        self.recording = False
        self.session = None
        self.writer: Optional[DiskWriter] = None

        # Stats
        self.lidar_count = 0
        self.camera_count = 0
        self.pose_count = 0
        self.start_time: Optional[float] = None
        self.end_time_duration: Optional[float] = None
        self.last_lidar_time = 0.0
        self.last_camera_time = 0.0

    def run(self) -> None:
        """Main worker thread loop."""
        try:
            # Connect to robot
            logger.info(f"Connecting to {self.hostname}")
            self._connect_to_robot()

            self.running = True
            self.connected.emit()
            logger.info("Successfully connected to robot")

            # Polling loop
            while self.running:
                try:
                    if self.recording:
                        self._capture_frame()
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Frame capture error: {e}")
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"Capture worker error: {e}", exc_info=True)
            self.error.emit(str(e))
        finally:
            self.running = False
            self._cleanup()
            self.disconnected.emit()

    def _connect_to_robot(self) -> None:
        """Authenticate and initialize bosdyn SDK clients."""
        try:
            # Create SDK
            self.sdk = bosdyn.client.create_standard_sdk("spot-3d-reconstruction")

            # Create robot instance
            self.robot = self.sdk.create_robot(self.hostname)

            # Authenticate
            self.robot.authenticate(self.username, self.password)
            logger.info("Robot authentication successful")

            # Verify we can reach the robot
            self.robot.sync_with_directory()
            logger.info("Robot time sync complete")

            # Create client wrappers (these handle the bosdyn clients internally)
            try:
                lidar_client = self.robot.ensure_client('velodyne-point-cloud')
                self.lidar_client_wrapper = LidarClientWrapper(lidar_client)
                logger.info("LiDAR client initialized")
            except Exception as e:
                logger.warning(f"LiDAR client failed: {e}")

            try:
                image_client = self.robot.ensure_client(ImageClient.default_service_name)
                self.image_client_wrapper = ImageClientWrapper(image_client)
                logger.info("Image client initialized")
            except Exception as e:
                logger.warning(f"Image client failed: {e}")

            try:
                state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
                self.state_client_wrapper = StateClientWrapper(state_client)
                logger.info("State client initialized")
            except Exception as e:
                logger.warning(f"State client failed: {e}")

        except Exception as e:
            logger.error(f"Robot connection failed: {e}")
            raise

    def _capture_frame(self) -> None:
        """Capture a frame of LiDAR, images, and poses."""
        current_time = time.time()

        # Capture LiDAR (based on rate limit)
        if (current_time - self.last_lidar_time) > (1.0 / max(self.lidar_rate_hz, 0.1)):
            if self.lidar_client_wrapper:
                try:
                    lidar_frame = self.lidar_client_wrapper.get_scan()
                    if lidar_frame:
                        self.writer.write_lidar_frame(lidar_frame)
                        self.lidar_count += 1
                        self.last_lidar_time = current_time
                except Exception as e:
                    logger.debug(f"LiDAR capture failed: {e}")

        # Capture images (based on rate limit)
        if (current_time - self.last_camera_time) > (1.0 / max(self.camera_rate_hz, 0.1)):
            if self.image_client_wrapper:
                try:
                    camera_frames = self.image_client_wrapper.get_images()
                    for frame in camera_frames:
                        self.writer.write_camera_frame(frame)
                        self.camera_count += 1
                    self.last_camera_time = current_time
                except Exception as e:
                    logger.debug(f"Camera capture failed: {e}")

        # Capture pose (continuous, interpolated to LiDAR time)
        if self.state_client_wrapper:
            try:
                pose = self.state_client_wrapper.get_robot_pose()
                if pose:
                    self.writer.write_pose(pose)
                    self.pose_count += 1
            except Exception as e:
                logger.debug(f"Pose capture failed: {e}")

        # Small sleep to prevent busy-waiting
        time.sleep(0.01)

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.recording:
            self.stop_recording()

        if self.robot:
            try:
                logger.info("Disconnecting from robot")
                # Robot context manager handles cleanup
            except Exception as e:
                logger.warning(f"Robot cleanup error: {e}")

    @Slot()
    def start_recording(self) -> None:
        """Start recording session."""
        if not self.running:
            self.error.emit("Not connected to robot")
            logger.warning("Not connected")
            return

        try:
            logger.info("Starting recording session")
            self.session = create_session(
                self.output_dir,
                robot_hostname=self.hostname,
                robot_username=self.username,
            )
            self.writer = DiskWriter(self.session)
            self.writer.start()

            # Save intrinsics from image client if available
            if self.image_client_wrapper:
                intrinsics = {}
                try:
                    sources = self.image_client_wrapper.get_available_sources()
                    for source in sources:
                        intrinsics[source] = {
                            "fx": 1.0,
                            "fy": 1.0,
                            "cx": 0.0,
                            "cy": 0.0,
                        }
                    if intrinsics:
                        save_intrinsics(self.session.session_path, intrinsics)
                except Exception as e:
                    logger.warning(f"Could not save intrinsics: {e}")

            self.recording = True
            self.start_time = time.time()
            self.end_time_duration = None
            self.lidar_count = 0
            self.camera_count = 0
            self.pose_count = 0

            logger.info(f"Recording started: {self.session.session_path.name}")

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.error.emit(f"Failed to start recording: {e}")

    @Slot()
    def stop_recording(self) -> None:
        """Stop recording session."""
        if not self.recording:
            return
        
        self.end_time_duration = time.time() 

        try:
            logger.info("Stopping recording")
            self.recording = False

            if self.writer:
                self.writer.stop()

            if self.session:
                # Update metadata with final counts
                self.session.metadata.lidar_frame_count = self.lidar_count
                self.session.metadata.image_frame_count = self.camera_count
                self.session.metadata.pose_frame_count = self.pose_count
                self.session.metadata.end_time = datetime.now() 
                save_session_metadata(self.session)

            logger.info(
                f"Recording complete: {self.lidar_count} LiDAR, "
                f"{self.camera_count} camera, {self.pose_count} poses in "
                f"{self.session.session_path if self.session else 'unknown'}"
            )

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.error.emit(f"Error stopping recording: {e}")

    def get_stats(self) -> dict:
        """Get current capture statistics."""
        duration = 0.0
        if self.start_time and not self.end_time_duration:
            duration = time.time() - self.start_time
        elif self.end_time_duration:
            duration = self.end_time_duration - self.start_time

        return {
            "lidar_count": self.lidar_count,
            "camera_count": self.camera_count,
            "pose_count": self.pose_count,
            "duration_sec": duration,
            "session_id": self.session.session_path.name if self.session else "N/A",
            "recording": self.recording,
            "hostname": self.hostname,
        }
