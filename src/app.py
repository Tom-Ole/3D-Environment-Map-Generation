"""
app.py -  Main application to capture images from Spot and save in COLMAP format for 3D reconstruction.
"""

import time
import bosdyn.client
import bosdyn.client.util
from src.utils.get_images import get_image, GetImageOptions
from src.utils.colmap_wirter import ColmapWriter
import argparse
import keyboard
import signal

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

running = True

def handle_sigint(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT,  handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)


def main(args):
    
    sdk = bosdyn.client.create_standard_sdk("image_depth_plus_visual")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    logger.info(f"Connected to robot at {args.hostname}")
    
    image_options: GetImageOptions = GetImageOptions(
        output_path=args.output,
        image_sources=args.sources or [
            "frontleft_fisheye_image",
            "frontright_fisheye_image",
            "back_fisheye_image",
            "left_fisheye_image",
            "right_fisheye_image",
        ],
        auto_rotate=True,
        pixel_format="PIXEL_FORMAT_RGB_U8",
        save=True,
        show=args.show,
    )

    sparse_dir = Path(args.output) / "sparse" / "0"
    colmap_writer = ColmapWriter(sparse_dir)
    logger.info(f"COLMAP Sparse model will be written to {sparse_dir}")

    dt = 1.0 / args.rate
    frame_id = 0
    image_results: list = []
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5

    logger.info(f"Starting capture: rate = {args.rate:.1f} Hz, max_frames = {args.output}, sources = {image_options.image_sources}")

    while running:
        loop_start = time.time()
        frame_id += 1

        try:
            get_image(robot, image_options, f"{frame_id:05d}", image_results, colmap_writer)
            consecutive_failures = 0  # reset on success
        except Exception as e:
            consecutive_failures += 1 
            logger.warning(
                f"Frame {frame_id:05d} FAILED ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}",
            )
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error("Too many consecutive failures - aborting capture.")
                break


        elapsed = time.time() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            logger.debug(f"Frame {frame_id:05d} took {elapsed:.3f}s (budget: {dt:.3f}s) - consider lowering the --rate")

        logger.info(f"Capture finished. Frames captured: {frame_id} | Total images {len(image_results)}")
        logger.info(f"Output directory: {Path(args.output).resolve()}")
    

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
        "--sources", nargs="+", default=None,
        metavar="SOURCE",
        help="Camera source names to capture. Defaults to all five fisheye cameras.",
    )
    parser.add_argument(
        "--rate", type=float, default=3.0,
        help="Target capture rate in Hz. 2-4 Hz recommended for COLMAP.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=500,
        help="Stop after this many frames (also stop with Ctrl-C).",
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
 
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)


    main(args)
