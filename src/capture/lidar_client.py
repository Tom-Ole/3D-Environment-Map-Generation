"""Thin wrapper around bosdyn PointCloudClient for EAP Velodyne LiDAR."""

import logging
import time
from typing import Optional

import numpy as np
from bosdyn.api import point_cloud_pb2
from bosdyn.client.point_cloud import PointCloudClient

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
        # TODO: verify on robot the exact service name for EAP Velodyne
        self.service_name = "velodyne"
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

            # Request point cloud from service
            # TODO: verify on robot the exact request/response for EAP point clouds
            request = point_cloud_pb2.GetPointCloudRequest(
                point_cloud_sources=[self.service_name]
            )

            response = self.client.get_point_cloud(request)

            if not response.point_cloud:
                logger.warning("Empty point cloud response from LiDAR")
                return None

            point_cloud = response.point_cloud[0]

            # Parse point data from protobuf
            # TODO: verify on robot the exact data format and field layout
            points_list = []
            intensity_list = []

            # Point clouds in bosdyn are typically stored as serialized data
            # We need to deserialize based on the sensor's format
            if hasattr(point_cloud, "data"):
                # Raw binary data - interpret as Nx4 (x, y, z, intensity)
                data = np.frombuffer(point_cloud.data, dtype=np.float32)
                if len(data) % 4 == 0:
                    data = data.reshape(-1, 4)
                    points = data[:, :3]
                    intensity = data[:, 3]
                else:
                    logger.error(f"Unexpected point cloud data size: {len(data)}")
                    return None
            else:
                logger.warning("Could not parse point cloud data format")
                return None

            frame = LidarFrame(
                timestamp=timestamp,
                points=points.astype(np.float32),
                intensity=(intensity * 255).astype(np.uint8),
                frame_id=self.frame_count,
            )

            self.frame_count += 1
            logger.debug(
                f"Captured LiDAR frame {frame.frame_id}: {len(points)} points"
            )

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
            # TODO: verify on robot how to list available point cloud sources
            return [self.service_name]
        except Exception as e:
            logger.error(f"Failed to get available point cloud sources: {e}")
            return []
