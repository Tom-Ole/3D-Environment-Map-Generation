"""Thin wrapper around bosdyn RobotStateClient for poses and IMU."""

import logging
import time
from typing import Optional, Tuple

import numpy as np
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)
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

            state = self.client.get_robot_state()
            frame_tree = state.kinematic_state.transforms_snapshot

            # Try vision first (odometry-corrected), fall back to odom
            tform = get_a_tform_b(frame_tree, VISION_FRAME_NAME, BODY_FRAME_NAME)
            frame_id = VISION_FRAME_NAME
            if tform is None:
                tform = get_a_tform_b(frame_tree, ODOM_FRAME_NAME, BODY_FRAME_NAME)
                frame_id = ODOM_FRAME_NAME

            if tform is None:
                logger.warning("Could not find body pose in frame tree")
                return None

            position = np.array(
                [tform.position.x, tform.position.y, tform.position.z],
                dtype=np.float32,
            )

            # SE3Pose.rot is a Quat with x/y/z/w (scalar-last)
            quaternion = np.array(
                [tform.rot.x, tform.rot.y, tform.rot.z, tform.rot.w],
                dtype=np.float32,
            )

            pose = RobotPose(
                timestamp=timestamp,
                position=position,
                quaternion=quaternion,
                frame_id=frame_id,
            )

            logger.debug(f"Got robot pose in {frame_id}: {position}, quat: {quaternion}")
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
