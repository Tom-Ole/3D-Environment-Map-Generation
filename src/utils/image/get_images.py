# Reference:
#   https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/get_image/get_image.py
#   https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/xbox_controller

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy import ndimage
from google.protobuf.json_format import MessageToDict

from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import get_a_tform_b, get_vision_tform_body, VISION_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot import Robot
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.geometry import EulerZXY

from utils.image.ImageOptions import ImageOptions, ImageSources
from utils.image.spot_colmap import SpotColmapWriter, apply_inplane_rotation, cam_to_world_to_colmap_pose

import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-camera display rotation (degrees, positive = CCW)
# ---------------------------------------------------------------------------
# These values match the physical mounting orientation of each camera on Spot.
# ndimage.rotate(img, angle) is called when angle != 0 so the saved image
# appears correctly oriented.  apply_inplane_rotation() is ALSO called on
# cam_to_world with the same angle so the COLMAP extrinsic matches the
# displayed (rotated) image frame.
#
# Camera mounting summary (verified against BD robot geometry):
#   back, left     : sensor upright → no rotation needed
#   frontleft, frontright : sensor rotated 90° CW from upright → display needs -90°
#   right          : sensor upside-down → display needs 180°
ROTATION_ANGLE: Dict[str, float] = {
    "back_fisheye_image":       0,
    "frontleft_fisheye_image":  -90,
    "frontright_fisheye_image": -90,
    "left_fisheye_image":       0,
    "right_fisheye_image":      180,
}

# ---------------------------------------------------------------------------
# Body-roll tilt for side cameras (optional feature)
# ---------------------------------------------------------------------------
_SIDE_CAMERA_ROLL: Dict[str, float] = {
    "left_fisheye_image":  +1.0,
    "right_fisheye_image": -1.0,
}

# ---------------------------------------------------------------------------
# Nominal OPENCV_FISHEYE distortion coefficients
# ---------------------------------------------------------------------------
# Spot's image proto (PinholeModel.CameraIntrinsics) does not expose
# distortion coefficients — only focal length, principal point, and skew.
# The values below are nominal Kannala-Brandt (k1..k4) equidistant fisheye
# coefficients that match published BD calibration data and community
# measurements.  They are a better starting point than zeros but will differ
# per unit; run a checkerboard calibration for exact per-robot values.
_NOMINAL_DISTORTION: Dict[str, Dict[str, float]] = {
    "back_fisheye_image":       {"k1": -0.009, "k2": 0.00046, "k3": -0.000019, "k4": 0.0},
    "left_fisheye_image":       {"k1": -0.009, "k2": 0.00046, "k3": -0.000019, "k4": 0.0},
    "right_fisheye_image":      {"k1": -0.009, "k2": 0.00046, "k3": -0.000019, "k4": 0.0},
    "frontleft_fisheye_image":  {"k1": -0.009, "k2": 0.00046, "k3": -0.000019, "k4": 0.0},
    "frontright_fisheye_image": {"k1": -0.009, "k2": 0.00046, "k3": -0.000019, "k4": 0.0},
}


def get_distortion_coeffs(
    source_name: str,
    override: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Return fisheye distortion coefficients for *source_name*."""
    if override and source_name in override:
        return override[source_name]
    if source_name in _NOMINAL_DISTORTION:
        return _NOMINAL_DISTORTION[source_name]
    logger.warning(
        "No distortion coefficients for '%s' — k1-k4 will be 0. "
        "OPENCV_FISHEYE degenerates to PINHOLE for this camera.",
        source_name,
    )
    return {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0}


def se3_to_matrix(se3) -> np.ndarray:
    """Convert a Spot SE3Pose proto to a 4×4 homogeneous transform matrix."""
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
    """
    Recompute cx/cy after ndimage.rotate(reshape=True) resamples the image.

    All other intrinsic values (fx, fy, distortion) are unaffected by an
    in-plane rotation — only the principal-point pixel coordinates change
    because the image canvas size changes.
    """
    if angle_deg == 0:
        return dict(intrinsics)

    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # Principal point relative to image centre.
    cx0 = original_cols / 2.0
    cy0 = original_rows / 2.0
    px = intrinsics["cx"] - cx0
    py = intrinsics["cy"] - cy0

    # Rotate the principal-point offset.
    px_rot =  cos_a * px - sin_a * py
    py_rot =  sin_a * px + cos_a * py

    # ndimage.rotate(reshape=True) canvas dimensions.
    new_cols = int(np.round(abs(original_cols * cos_a) + abs(original_rows * sin_a)))
    new_rows = int(np.round(abs(original_cols * sin_a) + abs(original_rows * cos_a)))

    new_cx = px_rot + new_cols / 2.0
    new_cy = py_rot + new_rows / 2.0

    return {**intrinsics, "cx": new_cx, "cy": new_cy}


def pixel_format_num_bytes(pixel_format: int) -> int:
    return {
        image_pb2.Image.PIXEL_FORMAT_RGB_U8:        3,
        image_pb2.Image.PIXEL_FORMAT_RGBA_U8:       4,
        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:  1,
        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16: 2,
    }.get(pixel_format, 1)


def _command_body_tilt(
    robot: Robot,
    roll_deg: float,
    settle_time: float = 1.0,
    lease=None,
) -> None:
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    orientation = EulerZXY(yaw=0.0, roll=np.radians(roll_deg), pitch=0.0)
    cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=orientation)
    command_client.robot_command(cmd)
    time.sleep(settle_time)


def get_image(
    robot: Robot,
    image_client: ImageClient,
    robot_state_client: RobotStateClient,
    options: ImageOptions,
    frame_id: str,
    colmap_writer: SpotColmapWriter,
    distortion_override: Optional[Dict[str, Dict[str, float]]] = None,
    lease=None,
) -> None:
    """
    Capture one frame from every source in *options.sources*, save images and
    metadata to disk, and register each image with *colmap_writer*.

    Parameters
    ----------
    robot : Robot
        Connected, authenticated Spot robot handle.
    image_client : ImageClient
        Pre-constructed image service client.
    robot_state_client : RobotStateClient
        Pre-constructed robot-state service client.
    options : ImageOptions
        Capture configuration (sources, output_path, rotation, …).
    frame_id : str
        Zero-padded frame number used as the image filename stem, e.g. ``"00042"``.
    colmap_writer : SpotColmapWriter
        Writer instance to register cameras and images with.  Must remain open
        (not yet saved) for the lifetime of this call.
    distortion_override : dict, optional
        Per-robot OPENCV_FISHEYE k1–k4 keyed by source name.  Overrides the
        built-in nominal table when provided.
    lease : LeaseKeepAlive, optional
        Active lease; passed to body-tilt commands when ``options.side_tilt`` is
        enabled.

    Notes
    -----
    Pose pipeline per image:

    1. ``get_a_tform_b(snapshot, "vision", frame_name_image_sensor)``
       returns ``T_vision_camera`` — the camera-to-world transform in Spot's
       vision frame (X-forward, Y-left, Z-up, right-handed).

    2. ``apply_inplane_rotation(cam_to_world, rotation_deg)`` composes
       ``Rz(-rotation_deg)`` onto cam_to_world so the COLMAP extrinsic aligns
       with the saved (display-rotated) image frame.

    3. ``SpotColmapWriter.add_image()`` inverts the result to world-to-camera
       and writes the Hamilton quaternion + translation to images.txt.
    """
    if not options.sources:
        raise ValueError("get_image: no sources specified in ImageOptions")

    # ------------------------------------------------------------------ capture
    robot_state = robot_state_client.get_robot_state()
    image_responses: List = []

    try:
        for source in options.sources:
            source_str = source.value
            if options.side_tilt and source_str in _SIDE_CAMERA_ROLL:
                roll_deg = options.side_tile_angle * _SIDE_CAMERA_ROLL[source_str]
                logger.info("Tilting body %.1f° for %s", roll_deg, source_str)
                _command_body_tilt(robot, roll_deg, options.tilt_settle_time, lease)
                req = build_image_request(
                    source_str,
                    quality_percent=100,
                    pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                )
                image_responses.extend(image_client.get_image([req]))
                _command_body_tilt(robot, 0, options.tilt_settle_time, lease)
            else:
                req = build_image_request(
                    source_str,
                    quality_percent=100,
                    pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                )
                image_responses.extend(image_client.get_image([req]))
    except Exception as exc:
        logger.error("Image capture failed: %s", exc)
        _command_body_tilt(robot, 0, options.tilt_settle_time, lease)
        raise

    # ---------------------------------------------------------------- output dirs
    base_dir       = Path(options.output_path) / "images"
    images_dir     = base_dir / "images"
    metadata_dir   = base_dir / "extra" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- per-image
    for image_response in image_responses:
        source_name = image_response.source.name

        # ---- decode ----
        is_depth  = image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
        dtype     = np.uint16 if is_depth else np.uint8
        extension = ".png"    if is_depth else ".jpg"
        num_bytes = 1 if is_depth else pixel_format_num_bytes(
            image_response.shot.image.pixel_format
        )

        raw = np.frombuffer(image_response.shot.image.data, dtype=dtype)
        if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                img = raw.reshape((
                    image_response.shot.image.rows,
                    image_response.shot.image.cols,
                    num_bytes,
                ))
            except ValueError:
                img = cv2.imdecode(raw, -1)
        else:
            img = cv2.imdecode(raw, -1)

        original_rows = image_response.shot.image.rows
        original_cols = image_response.shot.image.cols

        # ---- display rotation ----
        rotation_applied = 0.0
        if options.correct_image_rotation:
            if source_name in ROTATION_ANGLE:
                rotation_applied = ROTATION_ANGLE[source_name]
                if rotation_applied != 0:
                    img = ndimage.rotate(img, rotation_applied)
            else:
                logger.warning("No ROTATION_ANGLE entry for source '%s'", source_name)

        # ---- intrinsics ----
        intrinsics_data: Optional[Dict] = None
        if image_response.source.HasField("pinhole"):
            intr = image_response.source.pinhole.intrinsics
            # Spot SDK already provides principal_point in the *display-oriented* frame.
            # For cameras with a non-zero ROTATION_ANGLE (frontleft, frontright, right),
            # the SDK reports cx/cy as coordinates in the rotated image, not the raw
            # sensor readout.  Do NOT call adjust_intrinsics_for_rotation here — doing
            # so would rotate already-correct values and produce a wrong principal point.
            # Width/height come from img.shape[:2] *after* ndimage.rotate, so they match
            # the saved file regardless of whether a rotation was applied.
            intrinsics_data = {
                "fx": float(intr.focal_length.x),
                "fy": float(intr.focal_length.y),
                "cx": float(intr.principal_point.x),
                "cy": float(intr.principal_point.y),
            }
            intrinsics_data.update(get_distortion_coeffs(source_name, distortion_override))
        else:
            logger.warning("Source '%s' has no pinhole intrinsics — skipping COLMAP registration", source_name)

        # ---- pose ----
        snapshot    = image_response.shot.transforms_snapshot
        frame_name  = image_response.shot.frame_name_image_sensor

        logger.info(
            "Pose lookup: source='%s'  frame_name_image_sensor='%s'",
            source_name, frame_name,
        )

        vision_T_camera = get_a_tform_b(snapshot, VISION_FRAME_NAME, frame_name)
        if vision_T_camera is None:
            # Try the alternate name without the "_image" suffix (SDK version compat).
            alt_frame = frame_name.replace("_image", "") if frame_name.endswith("_image") else frame_name
            logger.warning(
                "get_a_tform_b('%s', '%s') returned None — retrying with '%s'",
                VISION_FRAME_NAME, frame_name, alt_frame,
            )
            vision_T_camera = get_a_tform_b(snapshot, VISION_FRAME_NAME, alt_frame)
            if vision_T_camera is None:
                logger.error(
                    "Cannot obtain pose for '%s' (tried '%s' and '%s') — skipping COLMAP registration",
                    source_name, frame_name, alt_frame,
                )
                if options.save:
                    cam_images_dir = images_dir / source_name
                    cam_images_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(cam_images_dir / f"{frame_id}{extension}"), img)
                continue

        cam_to_world = se3_to_matrix(vision_T_camera)

        logger.info(
            "cam_to_world for '%s':\n%s",
            source_name,
            np.array2string(cam_to_world, precision=4, suppress_small=True),
        )

        # Apply in-plane rotation compensation so the COLMAP extrinsic matches
        # the displayed (rotation-corrected) image frame.
        cam_to_world_colmap = apply_inplane_rotation(cam_to_world, rotation_applied)

        if rotation_applied != 0:
            logger.info(
                "Applied inplane rotation %.1f° to cam_to_world for '%s'",
                rotation_applied, source_name,
            )

        # ---- robot body pose (metadata only) ----
        try:
            vision_T_body    = get_vision_tform_body(
                robot_state.kinematic_state.transforms_snapshot
            )
            body_pose_matrix = se3_to_matrix(vision_T_body)
        except Exception as exc:
            logger.warning("Could not get body pose: %s", exc)
            body_pose_matrix = np.eye(4)

        vel = robot_state.kinematic_state.velocity_of_body_in_vision
        velocity_data = {
            "linear":  {"x": vel.linear.x,  "y": vel.linear.y,  "z": vel.linear.z},
            "angular": {"x": vel.angular.x, "y": vel.angular.y, "z": vel.angular.z},
        }

        # ---- file paths ----
        cam_images_dir = images_dir / source_name
        cam_images_dir.mkdir(parents=True, exist_ok=True)

        image_filename  = f"{frame_id}{extension}"
        image_rel_path  = f"{source_name}/{image_filename}"
        image_save_path = cam_images_dir / image_filename
        meta_save_path  = metadata_dir / f"{frame_id}_{source_name}.json"

        # ---- COLMAP registration ----
        if intrinsics_data is not None:
            h, w = img.shape[:2]
            camera_id = colmap_writer.register_camera(
                source_name,
                int(w),
                int(h),
                intrinsics_data,
            )
            colmap_writer.add_image(image_rel_path, camera_id, cam_to_world_colmap)

        # ---- save image ----
        if options.save:
            cv2.imwrite(str(image_save_path), img)

        # ---- metadata ----
        colmap_qwxyz_txyz = cam_to_world_to_colmap_pose(cam_to_world_colmap)
        metadata = {
            "frame_id":             frame_id,
            "source":               source_name,
            "frame_name_sensor":    frame_name,
            "rows":                 original_rows,
            "cols":                 original_cols,
            "rows_after_rot":       img.shape[0],
            "cols_after_rot":       img.shape[1],
            "rotation_applied_deg": rotation_applied,
            "timestamp":            MessageToDict(image_response.shot.acquisition_time),
            "intrinsics":           intrinsics_data,
            # Raw Spot vision-frame cam-to-world (preserved for traceability).
            "camera_to_world_spot": cam_to_world.tolist(),
            # Rotation-compensated cam-to-world actually used for COLMAP.
            "camera_to_world_colmap_input": cam_to_world_colmap.tolist(),
            # COLMAP world-to-camera pose (matches images.txt exactly).
            "colmap_pose": dict(zip(
                ["qw", "qx", "qy", "qz", "tx", "ty", "tz"],
                colmap_qwxyz_txyz,
            )),
            "robot_pose":     body_pose_matrix.tolist(),
            "robot_velocity": velocity_data,
        }

        if options.save:
            with open(meta_save_path, "w") as fh:
                json.dump(metadata, fh, indent=2)

        if options.show:
            cv2.imshow(source_name, img)

    if options.show:
        cv2.waitKey(0)
