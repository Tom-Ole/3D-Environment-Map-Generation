"""Thin wrapper around bosdyn PointCloudClient for EAP Velodyne LiDAR."""

import logging
import time
from typing import Optional

import numpy as np
from bosdyn.client.point_cloud import PointCloudClient, build_pc_request

from capture.types import LidarFrame

logger = logging.getLogger(__name__)


class LidarClientWrapper:
    """
    Wrapper around PointCloudClient for reading Velodyne-16 LiDAR from EAP.

    TODO: verify on robot that service_name="velodyne" is correct.
    """

    def __init__(self, client: PointCloudClient):
        """
        Initialize LiDAR client.

        Args:
            client: bosdyn.client.point_cloud.PointCloudClient
        """
        self.client = client
        self.source_name = "velodyne-point-cloud"
        self.frame_count = 0

    def get_scan(self) -> Optional[LidarFrame]:
        """
        Capture a single LiDAR scan from the Velodyne sensor.

        Returns:
            LidarFrame with point cloud data, or None if capture failed.

        Raises:
            RpcError: If communication with robot fails.
        """
        try:
            timestamp = time.time()

            # get_point_cloud returns a list of PointCloudResponse
            responses = self.client.get_point_cloud([build_pc_request(self.source_name)])

            if not responses or not responses[0].point_cloud.data:
                logger.warning("Empty point cloud response from LiDAR")
                return None

            point_cloud = responses[0].point_cloud

            # Velodyne data is interleaved float32 XYZ (3 floats per point, no intensity)
            data = np.frombuffer(point_cloud.data, dtype=np.float32)
            if len(data) % 3 != 0:
                logger.error(f"Unexpected point cloud data length: {len(data)}")
                return None

            points = data.reshape(-1, 3)
            intensity = np.zeros(len(points), dtype=np.uint8)

            frame = LidarFrame(
                timestamp=timestamp,
                points=points,
                intensity=intensity,
                frame_id=self.frame_count,
            )

            self.frame_count += 1
            logger.debug(f"Captured LiDAR frame {frame.frame_id}: {len(points)} points")

            return frame

        except Exception as e:
            logger.error(f"Failed to capture LiDAR scan: {e}")
            return None

    def get_available_sources(self) -> list:
        """
        Get list of available point cloud sources.

        Returns:
            List of source names (typically includes "velodyne")
        """
        try:
            sources = self.client.list_point_cloud_sources()
            return [s.name for s in sources]
        except Exception as e:
            logger.error(f"Failed to get available point cloud sources: {e}")
            return []
