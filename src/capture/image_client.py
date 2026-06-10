"""Thin wrapper around bosdyn ImageClient for body cameras."""

import logging
import time
from typing import List, Optional

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

from capture.types import CameraFrame

logger = logging.getLogger(__name__)

# cv2.rotate() code per camera, None = no rotation
CAMERA_ROTATION = {
    "back_fisheye_image": None,
    "left_fisheye_image": None,
    "right_fisheye_image": cv2.ROTATE_180,
    "frontleft_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
    "frontright_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
}

# Standard SPOT body camera source names
DEFAULT_CAMERA_SOURCES = [
    "back_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
]


class ImageClientWrapper:
    """
    Wrapper around ImageClient for reading SPOT body cameras.

    Captures images from 5 fisheye cameras with distortion models.
    """

    def __init__(
        self,
        client: ImageClient,
        sources: Optional[List[str]] = None,
    ):
        """
        Initialize image client.

        Args:
            client: bosdyn.client.image.ImageClient
            sources: List of camera source names (default: all 5 fisheyes)
        """
        self.client = client
        self.sources = sources or DEFAULT_CAMERA_SOURCES
        self.frame_counts = {src: 0 for src in self.sources}

    def get_images(self) -> List[CameraFrame]:
        """
        Capture images from all configured sources.

        Returns:
            List of CameraFrame objects (one per source)

        Raises:
            RpcError: If communication with robot fails.
        """
        frames = []
        timestamp = time.time()

        try:
            image_responses = self.client.get_image(
                [
                    build_image_request(src, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)
                    for src in self.sources
                ]
            )

            # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#bosdyn-api-ImageResponse
            for response in image_responses:
                source_name = response.source.name

                # Parse image data based on format
                # ImageResponse structure: response.shot.image.{data,format,rows,cols}
                img_proto = response.shot.image
                raw = np.frombuffer(img_proto.data, dtype=np.uint8)

                if img_proto.format == image_pb2.Image.FORMAT_RAW:
                    width, height = img_proto.cols, img_proto.rows
                    pf = img_proto.pixel_format
                    if pf == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                        image_data = raw.reshape(height, width, 3)[..., ::-1]  # RGB→BGR
                    elif pf == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                        image_data = cv2.cvtColor(
                            raw.reshape(height, width, 4), cv2.COLOR_RGBA2BGR
                        )
                    elif pf in (
                        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                        image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16,
                    ):
                        image_data = cv2.cvtColor(
                            raw.reshape(height, width), cv2.COLOR_GRAY2BGR
                        )
                    else:
                        # Unknown format — try imdecode as fallback
                        image_data = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                else:
                    # FORMAT_JPEG and any other encoded format
                    image_data = cv2.imdecode(raw, cv2.IMREAD_COLOR)

                if image_data is None:
                    logger.warning(f"Failed to decode image from {source_name}")
                    continue

                rotation = CAMERA_ROTATION.get(source_name)
                if rotation is not None:
                    image_data = cv2.rotate(image_data, rotation)

                # Extract intrinsics if available
                fx, fy, cx, cy = None, None, None, None
                distortion = None

                # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#bosdyn-api-ImageSource
                if response.source.pinhole and False:
                    fx = response.source.pinhole.intrinsics.focal_length.x
                    fy = response.source.pinhole.intrinsics.focal_length.y
                    cx = response.source.pinhole.intrinsics.principal_point.x
                    cy = response.source.pinhole.intrinsics.principal_point.y
                elif response.source.kannala_brandt:
                    # TODO: verify on robot the exact kannala_brandt model and parameters
                    logger.debug("FISHEYE Model found")
                    fx = response.source.kannala_brandt.intrinsics.pinhole_intrinsics.focal_length.x
                    fy = response.source.kannala_brandt.intrinsics.pinhole_intrinsics.focal_length.y
                    cx = response.source.kannala_brandt.intrinsics.pinhole_intrinsics.principal_point.x
                    cy = response.source.kannala_brandt.intrinsics.pinhole_intrinsics.principal_point.y

                    # Kannala-Brandt fisheye distortion parameters
                    distortion = {
                        "model": "kannala_brandt",
                        "k1": response.source.kannala_brandt.intrinsics.k1,
                        "k2": response.source.kannala_brandt.intrinsics.k2,
                        "k3": response.source.kannala_brandt.intrinsics.k3,
                        "k4": response.source.kannala_brandt.intrinsics.k4,
                    }

                frame = CameraFrame(
                    timestamp=timestamp,
                    source_name=source_name,
                    image_data=image_data,
                    frame_id=self.frame_counts[source_name],
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    distortion=distortion,
                )

                frames.append(frame)
                self.frame_counts[source_name] += 1

        except Exception as e:
            logger.error(f"Failed to capture images: {e}")

        logger.debug(f"Captured {len(frames)} images at timestamp {timestamp}")
        return frames

    def get_available_sources(self) -> List[str]:
        """
        Get list of available camera sources.

        Returns:
            List of source names
        """
        try:
            image_responses = self.client.get_image(
                [build_image_request(src) for src in self.sources]
            )
            return [resp.source.name for resp in image_responses]
        except Exception as e:
            logger.error(f"Failed to get available image sources: {e}")
            return []
