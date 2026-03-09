
from dataclasses import dataclass
from typing import List, Dict
import cv2
import numpy as np
from scipy import ndimage

from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.robot import Robot
from pathlib import Path
import json
from google.protobuf.json_format import MessageToDict

@dataclass
class GetImageOptions:
    output_path: str = "./"
    image_service: str = ImageClient.default_service_name

    image_sources: list[str] | None = None

    list: bool = False
    auto_rotate: bool = True

    pixel_format: str = "PIXEL_FORMAT_RGB_U8"

    show: bool = False
    save: bool = True


ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)

def get_image_sources(image_client):
    image_sources = image_client.list_image_sources()
    print('Image sources:')
    for source in image_sources:
        print('\t' + source.name)


def get_image(robot: Robot, options: GetImageOptions) -> List[Dict]:

    image_client = robot.ensure_client(options.image_service)

    if options.list:
        get_image_sources(image_client)
    
    if not options.image_sources:
        raise ValueError("No image_sources specified")

    # Capture and save images to disk
    pixel_format = pixel_format_string_to_enum(options.pixel_format)

    if pixel_format is None:
        raise ValueError(f"Invalid pixel format: {options.pixel_format}")

    image_request = [
        build_image_request(source, pixel_format=pixel_format)
        for source in options.image_sources
    ]
    image_responses = image_client.get_image(image_request)

    save_path = Path(options.output_path)
    save_path.mkdir(parents=True, exist_ok=True)

    results = []

    for image in image_responses:
        num_bytes = 1  # Assume a default of 1 byte encodings.
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
            extension = '.png'
        else:
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                num_bytes = 3
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                num_bytes = 4
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                num_bytes = 1
            elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                num_bytes = 2
            dtype = np.uint8
            extension = '.jpg'

        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                # Attempt to reshape array into an RGB rows X cols shape.
                img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
            except ValueError:
                # Unable to reshape the image data, trying a regular decode.
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        if options.auto_rotate and image.source.name in ROTATION_ANGLE:
            img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name]) # TODO: Maybe switch to OpenCV for better performance
        elif options.auto_rotate and not image.source.name in ROTATION_ANGLE:
            print(f"No rotation defined for source: {image.source.name}") # TODO: Maybe change to an logger

        timestamp = image.shot.acquisition_time
        filename = f"{image.source.name}_{timestamp.seconds}_{timestamp.nanos}"

        image_saved_path = save_path / filename.replace("/", "")


        # Save Metadata

        intrinsics_data = None


        if image.source.HasField("pinhole"):
            intrinsics = image.source.pinhole.intrinsics
            intrinsics_data = {
                "fx": intrinsics.focal_length.x,
                "fy": intrinsics.focal_length.y,
                "cx": intrinsics.principal_point.x,
                "cy": intrinsics.principal_point.y,
                "skew": intrinsics.skew
            }

        transform_snapshot = MessageToDict(image.shot.transforms_snapshot)

        # image.shot: https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#bosdyn-api-ImageCapture
        metadata = {
            "source": image.source.name,
            "rows": image.shot.image.rows,
            "cols": image.shot.image.cols,
            "timestamp": MessageToDict(timestamp),
            "intrinsics": intrinsics_data,
            "frame_name": image.shot.frame_name_image_sensor,
            "transform_snapshot": transform_snapshot
        }

        if options.save:
            cv2.imwrite(str(image_saved_path) + extension, img)

            with open(image_saved_path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2)

        # show image
        if options.show:
            cv2.imshow(image.source.name, img)

        results.append({
            "source": image.source.name,
            "image": img,
            "timestamp": timestamp,
            "path": str(image_saved_path)
        })

    if options.show:
        cv2.waitKey(0)

    return results