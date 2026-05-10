import math
import time
import bosdyn.client
import bosdyn.client.util
from utils.ImageOptions import ImageOptions, ImageSources
from utils.colmap_wirter import ColmapWriter
import argparse
import signal
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseKeepAlive

import logging
from pathlib import Path

from utils.get_images import get_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

running = True

def handle_sigint(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT,  handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

def _body_xy(robot_state_client: RobotStateClient) -> tuple[float, float]:
    """Get the robot's current (x, y) position in the odom frame."""
    robot_state = robot_state_client.get_robot_state()
    tform_odom_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
    return tform_odom_body.x, tform_odom_body.y

def _dist(a: tuple, b: tuple) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def list_image_sources(image_client) -> None:
    image_sources = image_client.list_image_sources()
    print("Image sources:")
    for source in image_sources:
        print("\t" + source.name)

def main(args):
    
    sdk = bosdyn.client.create_standard_sdk("image_depth_plus_visual")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    logger.info(f"Connected to robot at {args.hostname}")

    robot.time_sync.wait_for_sync()
    logger.info("Time sync established")


    image_options = ImageOptions(
        output_path=args.output,
        sources=ImageSources.get_color()
        )
    image_options.show = args.show
    

    sparse_dir = Path(args.output) / "sparse" / "0"
    with ColmapWriter(sparse_dir) as colmap_writer:
        logger.info(f"COLMAP Sparse model will be written to {sparse_dir}")

        # Clients
        lease_client       = robot.ensure_client(LeaseClient.default_service_name)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        image_client       = robot.ensure_client(ImageClient.default_service_name)
        command_client     = robot.ensure_client(RobotCommandClient.default_service_name)

        # state
        step_m = float(args.n)
        frame_count = 0
        last_pos = _body_xy(robot_state_client)

        logger.info(f"Tablet is in control. Walking {step_m:.2f} m between captures. Ctrl-C to stop.")

        while running:
            curr_pos = _body_xy(robot_state_client)
            walked = _dist(last_pos, curr_pos)

            if walked < step_m:
                time.sleep(0.1)
                continue

            logger.info(f"walked {walked:.2f} m - takeing lease for capture {frame_count:05d}")


            lease = lease_client.acquire()
            with LeaseKeepAlive(lease_client=lease_client, must_acquire=True, return_at_exit=True):
                blocking_stand(command_client, timeout_sec=10)

                frame_id = f"frame_{frame_count:05d}"

                get_image(robot=robot, image_client=image_client, robot_state_client=robot_state_client, image_options=image_options, frame_id= frame_id, colmap_writer=colmap_writer, lease=lease)

                frame_count += 1
                logger.info("Capture #%s done. Total frames: %d", frame_id, frame_count)
            
            last_pos = _body_xy(robot_state_client)



        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Capture images from Spot and save in COLMAP format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hostname", required=True,
        help="Robot hostname or IP, e.g. 192.168.80.3",
    )
    parser.add_argument(
        "--output", default="./output",
        help="Root directory for captured data.",
    )
    parser.add_argument(
        "--n", default=1,
        help="The distance between between captures in meters.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display images in OpenCV windows (requires a display).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level log messages.",
    )
 
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)


    main(args)
