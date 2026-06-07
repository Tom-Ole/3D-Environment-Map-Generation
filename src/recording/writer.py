"""Thread-safe disk writer for recording sessions."""

import json
import logging
import threading
from pathlib import Path
from queue import Queue
from typing import Optional

import numpy as np
from open3d.geometry import PointCloud
from open3d.io import write_point_cloud

from capture.types import CameraFrame, LidarFrame, RobotPose
from recording.session import save_intrinsics, save_poses, save_session_metadata
from recording.types import RecordingSession

logger = logging.getLogger(__name__)


class DiskWriter:
    """
    Thread-safe disk writer for recording session data.

    Runs in a background thread and processes write requests from a queue.
    """

    def __init__(self, session: RecordingSession):
        """
        Initialize disk writer.

        Args:
            session: RecordingSession to write to
        """
        self.session = session
        self.queue: Queue = Queue(maxsize=100)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Buffer for poses (accumulated and saved periodically)
        self.pose_buffer: list = []
        self.pose_lock = threading.Lock()

    def start(self) -> None:
        """Start the writer thread."""
        if self.running:
            logger.warning("Writer already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        logger.info("Disk writer started")

    def stop(self) -> None:
        """Stop the writer thread and flush remaining data."""
        if not self.running:
            return

        self.running = False

        # Flush remaining items
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                self._process_item(item)
            except:
                break

        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5.0)

        # Save final metadata and poses
        with self.pose_lock:
            if self.pose_buffer:
                poses_array = np.array(self.pose_buffer)
                save_poses(self.session.session_path, poses_array)
                self.pose_buffer.clear()

        save_session_metadata(self.session)
        logger.info("Disk writer stopped")

    def write_lidar_frame(self, frame: LidarFrame) -> None:
        """Queue a LiDAR frame for writing."""
        self.queue.put(("lidar", frame))

    def write_camera_frame(self, frame: CameraFrame) -> None:
        """Queue a camera frame for writing."""
        self.queue.put(("camera", frame))

    def write_pose(self, pose: RobotPose) -> None:
        """Queue a pose for writing (buffered)."""
        self.queue.put(("pose", pose))

    def _writer_loop(self) -> None:
        """Main writer loop (runs in background thread)."""
        try:
            while self.running:
                try:
                    item = self.queue.get(timeout=1.0)
                    self._process_item(item)
                except Exception as e:
                    if self.running:
                        logger.debug(f"Queue timeout or error: {e}")
                    break

        except Exception as e:
            logger.error(f"Writer loop crashed: {e}")
        finally:
            self.running = False

    def _process_item(self, item: tuple) -> None:
        """Process a single queued item."""
        item_type = item[0]

        try:
            if item_type == "lidar":
                self._write_lidar_frame(item[1])
            elif item_type == "camera":
                self._write_camera_frame(item[1])
            elif item_type == "pose":
                self._write_pose(item[1])
        except Exception as e:
            logger.error(f"Failed to write {item_type}: {e}")

    def _write_lidar_frame(self, frame: LidarFrame) -> None:
        """Write a LiDAR frame to disk as PLY."""
        lidar_path = self.session.get_lidar_folder()
        frame_path = lidar_path / f"{frame.frame_id:05d}.ply"

        # Create point cloud with intensity as color
        pcd = PointCloud()
        pcd.points.append(frame.points)

        # Store intensity as colors (repeat to RGB)
        intensity_normalized = frame.intensity.astype(np.float32) / 255.0
        colors = np.repeat(intensity_normalized[:, np.newaxis], 3, axis=1)
        pcd.colors.append(colors)

        write_point_cloud(str(frame_path), pcd)

        # Also save raw data
        raw_path = lidar_path / f"{frame.frame_id:05d}_raw.npy"
        raw_data = np.column_stack((frame.points, frame.intensity))
        np.save(raw_path, raw_data)

        self.session.metadata.lidar_frame_count += 1
        logger.debug(f"Wrote LiDAR frame {frame.frame_id} to {frame_path}")

    def _write_camera_frame(self, frame: CameraFrame) -> None:
        """Write a camera frame to disk as PNG."""
        import cv2

        images_path = self.session.get_image_folder()
        frame_path = (
            images_path / f"{frame.frame_id:05d}_{frame.source_name}.png"
        )

        # Convert BGR back to RGB for storage
        image_rgb = cv2.cvtColor(frame.image_data, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(frame_path), image_rgb)

        # Save intrinsics if available
        if frame.fx and frame.fy:
            if frame.source_name not in self.session.intrinsics:
                self.session.intrinsics[frame.source_name] = {
                    "fx": float(frame.fx),
                    "fy": float(frame.fy),
                    "cx": float(frame.cx) if frame.cx else 0.0,
                    "cy": float(frame.cy) if frame.cy else 0.0,
                    "distortion": frame.distortion or {},
                }

        self.session.metadata.image_frame_count += 1
        logger.debug(f"Wrote camera frame {frame.frame_id} ({frame.source_name})")

    def _write_pose(self, pose: RobotPose) -> None:
        """Buffer a pose for batch writing."""
        with self.pose_lock:
            # Store as [timestamp, x, y, z, qx, qy, qz, qw]
            pose_row = np.array(
                [
                    pose.timestamp,
                    pose.position[0],
                    pose.position[1],
                    pose.position[2],
                    pose.quaternion[0],
                    pose.quaternion[1],
                    pose.quaternion[2],
                    pose.quaternion[3],
                ],
                dtype=np.float32,
            )
            self.pose_buffer.append(pose_row)
            self.session.metadata.pose_frame_count += 1

            # Flush every 100 poses
            if len(self.pose_buffer) >= 100:
                poses_array = np.array(self.pose_buffer)
                save_poses(self.session.session_path, poses_array)
                self.pose_buffer.clear()

        logger.debug(f"Buffered pose: {pose.position}")
