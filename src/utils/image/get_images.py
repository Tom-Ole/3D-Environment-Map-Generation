# https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/get_image/get_image.py
# https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/xbox_controller
from dataclasses import dataclass
import logging
from typing import List, Dict, Optional
import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
import time

from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.robot import Robot, RobotCommandClient
from pathlib import Path
import json
from google.protobuf.json_format import MessageToDict
from bosdyn.client.frame_helpers import get_a_tform_b, get_vision_tform_body
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.geometry import EulerZXY
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive

from utils.image.ImageOptions import ImageOptions
from utils.image.colmap_wirter import ColmapWriter, matrix_to_colmap_pose

logger = logging.getLogger(__name__)


ROTATION_ANGLE = {
    "back_fisheye_image": 0,
    "frontleft_fisheye_image": -90, 
    "frontright_fisheye_image": -90,
    "left_fisheye_image": 0,
    "right_fisheye_image": 180
}

# Cameras that need a body roll to point away from the legs.
_SIDE_CAMERA_ROLL: dict[str, float] = {
    "left_fisheye_image":  +1.0,   # roll right -> left cam looks more upward
    "right_fisheye_image": -1.0,   # roll left -> right cam looks more upward
}

def se3_to_matrix(se3):
    """Convert a Spot SE3Pose proto to a 4x4 matrix"""
    mat = np.eye(4)
    mat[:3, :3] = se3.rotation.to_matrix()
    mat[:3, 3] = [se3.position.x, se3.position.y, se3.position.z]
    return mat

def adjust_intrinsics_for_rotation(
    intrinsics: dict,
    angle_deg: float,
    original_rows: int,
    original_cols: int,
) -> dict:
    if angle_deg == 0:
        return dict(intrinsics)
 
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
 
    # Centre of the original image
    cx0 = original_cols / 2.0
    cy0 = original_rows / 2.0
 
    # Principal point relative to image centre
    px = intrinsics["cx"] - cx0
    py = intrinsics["cy"] - cy0
 
    # Rotate the vector
    px_rot = cos_a * px - sin_a * py
    py_rot = sin_a * px + cos_a * py
 
    # ndimage.rotate(reshape=True) computes new canvas size
    new_cols = int(np.round(abs(original_cols * cos_a) + abs(original_rows * sin_a)))
    new_rows = int(np.round(abs(original_cols * sin_a) + abs(original_rows * cos_a)))
 
    new_cx = px_rot + new_cols / 2.0
    new_cy = py_rot + new_rows / 2.0
 
    return {**intrinsics, "cx": new_cx, "cy": new_cy}


def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]

def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)

def pixel_format_num_bytes(pixel_format) -> int:
    mapping = {
        image_pb2.Image.PIXEL_FORMAT_RGB_U8:        3,
        image_pb2.Image.PIXEL_FORMAT_RGBA_U8:       4,
        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:  1,
        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16: 2,
    }
    return mapping.get(pixel_format, 1)


def _command_body_tilt(robot: Robot, roll_deg: float, settle_time: float = 1.0, lease = None) -> None:

    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    # Clear any lingering behavior faults first
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_state = robot_state_client.get_robot_state()
    faults = robot_state.behavior_fault_state.faults
    if faults:
        logger.warning(f"Clearing {len(faults)} behavior fault(s) before tilt command")
        command_client.clear_behavior_fault(faults[0].behavior_fault_id)

    orientation = EulerZXY(yaw=0.0, roll=np.radians(roll_deg), pitch=0.0)
    cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=orientation)
    command_client.robot_command(cmd)
    time.sleep(settle_time)


def get_image(robot: Robot, image_client: ImageClient, robot_state_client: RobotStateClient, options: ImageOptions, frame_id: str, colmap_writer: ColmapWriter, lease = None) -> None:

    if not options.sources:
        raise ValueError("No image_sources specified in ImageOptions")

    robot_state = robot_state_client.get_robot_state()

    def take_image_request(source_name: str):
        return build_image_request(source_name, quality_percent=100, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)

    image_responses = []

    try:
        for source in options.sources:
            if options.side_tilt and source in _SIDE_CAMERA_ROLL:
                roll_deg = options.side_tile_angle * _SIDE_CAMERA_ROLL[source]
                logger.info(f"Tilting body by {roll_deg} deg for better {source} capture")
                _command_body_tilt(robot, roll_deg, options.tilt_settle_time, lease)
                request = take_image_request(source.value)
                image_responses.extend(image_client.get_image([request]))
                _command_body_tilt(robot, 0, options.tilt_settle_time, lease)
            else:
                request = take_image_request(source.value)
                image_responses.extend(image_client.get_image([request]))
    except Exception as e:
        logger.error(f"Error during side tilt: {e}")
        _command_body_tilt(robot, 0, options.tilt_settle_time, lease)
        raise e


    base = Path(options.output_path) / "images"
    images_dir = base / "images"
    metadata_dir = base / "extra" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for image in image_responses:

        source_name = image.source.name

        is_depth    = image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        dtype       = np.uint16 if is_depth else np.uint8
        extension   = ".png" if is_depth else ".jpg"
        num_bytes   = 1 if is_depth else pixel_format_num_bytes(image.shot.image.pixel_format)

        raw = np.frombuffer(image.shot.image.data, dtype=dtype)

        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                img = raw.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
            except ValueError:
                img = cv2.imdecode(raw, -1) # -1 = unchanged
        else:
            img = cv2.imdecode(raw, -1)

        original_rows = image.shot.image.rows
        original_cols = image.shot.image.cols

        rotation_applied = 0.0
        if options.correct_image_rotation:
            if source_name in ROTATION_ANGLE:
                rotation_applied = ROTATION_ANGLE[source_name]
                if rotation_applied != 0:
                    img = ndimage.rotate(img, rotation_applied)
            else:
                logger.warning(f"No rotation defined for source {source_name}")

        intrinsics_data: Optional[Dict] = None
        if image.source.HasField("pinhole"):
            intr = image.source.pinhole.intrinsics
            raw_intrinsics = {
                "fx"    : intr.focal_length.x,
                "fy"    : intr.focal_length.y,
                "cx"    : intr.principal_point.x,
                "cy"    : intr.principal_point.y,
                "skew"  : intr.skew.x,
                "skew_y"  : intr.skew.y,
            }
            intrinsics_data = adjust_intrinsics_for_rotation(
                raw_intrinsics, rotation_applied, original_rows, original_cols
            )

        snapshot     = image.shot.transforms_snapshot
        camera_frame = image.shot.frame_name_image_sensor

        vision_T_camera  = get_a_tform_b(snapshot, "vision", camera_frame)
        cam_to_world     = se3_to_matrix(vision_T_camera)

        vision_T_body    = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )
        body_pose_matrix = se3_to_matrix(vision_T_body)

        vel = robot_state.kinematic_state.velocity_of_body_in_vision
        velocity_data = {
            "linear":  {"x": vel.linear.x,  "y": vel.linear.y,  "z": vel.linear.z},
            "angular": {"x": vel.angular.x, "y": vel.angular.y, "z": vel.angular.z},
        }

        cam_images_dir = images_dir / source_name
        cam_images_dir.mkdir(parents=True, exist_ok=True)

        # e.g. "frontleft_fisheye_image/00042.jpg"
        image_filename  = f"{frame_id}{extension}"
        image_rel_path  = f"{source_name}/{image_filename}"
        image_save_path = cam_images_dir / image_filename
        meta_save_path  = metadata_dir / f"{frame_id}_{source_name}.json"

        if intrinsics_data is not None:
            h, w = img.shape[:2]
            camera_id = colmap_writer.register_camera(
                source_name, w, h, intrinsics_data
            )
            colmap_writer.write_image(image_rel_path, camera_id, cam_to_world)

        if options.save:
            cv2.imwrite(str(image_save_path), img)

        metadata = {
            "frame_id":             frame_id,
            "source":               source_name,
            "rows":                 original_rows,
            "cols":                 original_cols,
            "rows_after_rot":       img.shape[0],
            "cols_after_rot":       img.shape[1],
            "rotation_applied_deg": rotation_applied,
            "timestamp":            MessageToDict(image.shot.acquisition_time),
            "intrinsics":           intrinsics_data,
            "frame_name":           image.shot.frame_name_image_sensor,
            "camera_to_world":      cam_to_world.tolist(),
            "robot_pose":           body_pose_matrix.tolist(),
            "robot_velocity":       velocity_data,
            # Store the COLMAP-convention pose as well for convenience
            "colmap_pose": dict(zip(
                ["qw", "qx", "qy", "qz", "tx", "ty", "tz"],
                matrix_to_colmap_pose(cam_to_world),
            )),
        }

        if options.save:
            with open(meta_save_path, "w") as fh:
                json.dump(metadata, fh, indent=2)

        if options.show:
            cv2.imshow(source_name, img)

    if options.show:
        cv2.waitKey(0) # blocks (1 for non-blocking)