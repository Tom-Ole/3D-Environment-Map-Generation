"""Thin wrapper around bosdyn RobotStateClient for poses and IMU."""

import logging
import time
from typing import Optional, Tuple

import numpy as np
from bosdyn.client.robot_state import RobotStateClient

from capture.types import ImuData, RobotPose

logger = logging.getLogger(__name__)


class StateClientWrapper:
    """
    Wrapper around RobotStateClient for robot pose and IMU data.

    Extracts pose from frame tree and IMU from kinematic state.
    """

    def __init__(self, client: RobotStateClient):
        """
        Initialize state client.

        Args:
            client: bosdyn.client.robot_state.RobotStateClient
        """
        self.client = client
        self.frame_name = "vision"  # TODO: verify on robot (typically "vision" or "odom")

    def get_robot_pose(self) -> Optional[RobotPose]:
        """
        Get current robot body pose in the vision frame.

        Returns:
            RobotPose with position and orientation, or None if unavailable

        Raises:
            RpcError: If communication with robot fails.
        """
        try:
            timestamp = time.time()

            # Get frame tree snapshot
            state = self.client.get_robot_state()
            frame_tree = state.kinematic_state.transforms_snapshot

            # Look for body_tform_vision or odom_tform_body
            # TODO: verify on robot the exact frame names in the frame tree
            target_frame = f"{self.frame_name}_tform_body"

            body_tform = None
            for frame in frame_tree.child_to_parent_edge_map.values():
                if frame.parent_frame_name == self.frame_name:
                    body_tform = frame.parent_tform_child.inverse()
                    break

            if not body_tform:
                # Try alternative name
                for frame in frame_tree.child_to_parent_edge_map.values():
                    if frame.child_frame_name == self.frame_name:
                        body_tform = frame.parent_tform_child
                        break

            if not body_tform:
                logger.warning(f"Could not find {target_frame} in frame tree")
                return None

            # Extract position
            position = np.array(
                [
                    body_tform.position.x,
                    body_tform.position.y,
                    body_tform.position.z,
                ],
                dtype=np.float32,
            )

            # Extract quaternion (scalar-last convention)
            quaternion = np.array(
                [
                    body_tform.rotation.x,
                    body_tform.rotation.y,
                    body_tform.rotation.z,
                    body_tform.rotation.w,
                ],
                dtype=np.float32,
            )

            pose = RobotPose(
                timestamp=timestamp,
                position=position,
                quaternion=quaternion,
                frame_id=self.frame_name,
            )

            logger.debug(f"Got robot pose: {position}, quat: {quaternion}")
            return pose

        except Exception as e:
            logger.error(f"Failed to get robot pose: {e}")
            return None

    def get_imu_data(self) -> Optional[ImuData]:
        """
        Get IMU measurements from robot body.

        Returns:
            ImuData with acceleration and angular velocity, or None if unavailable

        Raises:
            RpcError: If communication with robot fails.
        """
        try:
            timestamp = time.time()

            state = self.client.get_robot_state()
            kinematic_state = state.kinematic_state

            if not kinematic_state.imu_data:
                logger.warning("No IMU data in robot state")
                return None

            imu = kinematic_state.imu_data

            # Linear acceleration
            linear_accel = np.array(
                [
                    imu.linear_acceleration.x,
                    imu.linear_acceleration.y,
                    imu.linear_acceleration.z,
                ],
                dtype=np.float32,
            )

            # Angular velocity
            angular_vel = np.array(
                [
                    imu.angular_velocity.x,
                    imu.angular_velocity.y,
                    imu.angular_velocity.z,
                ],
                dtype=np.float32,
            )

            imu_data = ImuData(
                timestamp=timestamp,
                linear_acceleration=linear_accel,
                angular_velocity=angular_vel,
            )

            return imu_data

        except Exception as e:
            logger.error(f"Failed to get IMU data: {e}")
            return None

    def get_robot_status(self) -> dict:
        """
        Get summary of robot status.

        Returns:
            Dictionary with connection, battery, motor state, and fault info
        """
        try:
            state = self.client.get_robot_state()

            status = {
                "power_state": state.power_state.motor_power_state,
                "battery_percentage": state.battery_states[0].charge_percentage
                if state.battery_states
                else 0.0,
                "system_fault_count": len(state.system_fault_state.faults),
                "estop_status": "unknown",
                "connected": True,
            }

            return status

        except Exception as e:
            logger.error(f"Failed to get robot status: {e}")
            return {"connected": False}
