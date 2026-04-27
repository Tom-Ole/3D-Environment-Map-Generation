"""
run_route.py - Upload a pre-recorded GraphNav map, navigate Spot autonomously
along the route, and capture images at every registered capture waypoint.

Prerequisites
---------------
1.  A route recorded with record_route.py (route_dir/route.json + map files).
2.  The robot must be placed at (or very near) the *seed waypoint* - the first
    position marked during recording.
    A recocnizable landmark (e.g. a fiducial) near the seed waypoint can help with localisation.

Usage
-------
python run_route.py --hostname 192.168.80.3 --route-dir ./routes/lab_loop \
    [--output ./output/lab_loop_001] \
    [--capture-rate 2.0]            # Hz during the brief stop at each waypoint
    [--capture-sources frontleft_fisheye_image frontright_fisheye_image]
    [--nav-velocity 0.8]            # m/s during transit (0 = SDK default)
    [--dry-run]                     # navigate but do not save images
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path
from typing import Optional

import bosdyn.client
import bosdyn.client.util
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import geometry_pb2

from utils.get_images import ColmapWriter, GetImageOptions, get_image
from utils.route import RouteDefinition, CaptureWaypoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info("Signal received - will stop after current waypoint.")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)



def upload_map(graph_nav_client: GraphNavClient, route_dir: Path): # -> map_pb2.Graph TODO: Fix type annotation
    """
    Upload the graph topology and all waypoint snapshots from *route_dir* to
    the robot.  Returns the deserialized Graph proto.
    """
    logger.info("Uploading map from %s ...", route_dir)

    graph_path = route_dir / "graph"
    with open(graph_path, "rb") as fh:
        graph = map_pb2.Graph()
        graph.ParseFromString(fh.read())

    graph_nav_client.upload_graph(graph=graph)
    logger.info("Graph uploaded  (%d waypoints, %d edges)",
                len(graph.waypoints), len(graph.edges))


    snapshots_dir = route_dir / "waypoint_snapshots"
    uploaded = 0
    for waypoint in graph.waypoints:
        snap_path = snapshots_dir / waypoint.snapshot_id
        if not snap_path.exists():
            logger.warning("Snapshot missing: %s", snap_path)
            continue
        with open(snap_path, "rb") as fh:
            snapshot = map_pb2.WaypointSnapshot()
            snapshot.ParseFromString(fh.read())
        graph_nav_client.upload_waypoint_snapshot(snapshot)
        uploaded += 1

    logger.info("Uploaded %d / %d waypoint snapshots.", uploaded, len(graph.waypoints))
    return graph



def localise_robot(
    graph_nav_client: GraphNavClient,
    robot_state_client,
    seed_waypoint_id: str,
    timeout_sec: float = 30.0,
) -> bool:
    """
    Attempt to localise the robot against the uploaded map.

    Strategy: use SetLocalization with the fiducial-based method first.
    Falls back to the nearest-waypoint hint if no fiducial is detected.

    The robot must be standing near *seed_waypoint_id*.

    Returns True on success, False on failure.
    """
    logger.info(f"Attempting localisation (seed waypoint: {seed_waypoint_id}) ...",)

    init_guess = nav_pb2.Localization()
    init_guess.waypoint_id = seed_waypoint_id

    try:
        graph_nav_client.set_localization(
            initial_guess_localization=init_guess,
            ko_tform_body=None,
            max_distance=3.0,
            max_yaw=1.57,   # ±90°
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST,
        )
    except Exception as exc:
        logger.warning(f"Fiducial localisation failed ({exc}). Trying waypoint hint...")
        try:
            robot_state = robot_state_client.get_robot_state()
            ko_tform_body = robot_state.kinematic_state.transform_snapshot
            graph_nav_client.set_localization(
                initial_guess_localization=init_guess,
                ko_tform_body=ko_tform_body,
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            )
        except Exception as exc2:
            logger.error(f"Localisation failed: {exc2}")
            return False

    # Verify localisation was accepted
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        state = graph_nav_client.get_localization_state()
        if state.localization.waypoint_id:
            logger.info(f"Localised at waypoint {state.localization.waypoint_id}")
            return True
        time.sleep(0.5)

    logger.error("Timed out waiting for localisation.")
    return False


def navigate_to_waypoint(
    graph_nav_client: GraphNavClient,
    waypoint_id: str,
    speed_limit: float = 0.0,
    timeout_sec: float = 60.0,
) -> bool:
    """
    Command the robot to navigate to *waypoint_id* and block until it arrives
    or the operation times out / fails.

    Parameters
    ----------
    speed_limit : Maximum travel speed in m/s.  0 = use SDK default.

    Returns True if the robot reached the waypoint, False otherwise.
    """


    nav_params = None
    if speed_limit > 0:
        nav_params = graph_nav_pb2.TravelParams(
            max_distance=0.0,          # walk to the waypoint exactly
            velocity_limit=geometry_pb2.SE2VelocityLimit(
                max_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=speed_limit, y=0),
                    angular=0,
                )
            ),
        )

    try:
        nav_to_cmd_id = graph_nav_client.navigate_to(
            waypoint_id,
            travel_params=nav_params,
            cmd_duration=100,
        )
    except Exception as exc:
        logger.error("navigate_to(%s) call failed: %s", waypoint_id, exc)
        return False

    STILL_NAVIGATING = {
    graph_nav_pb2.NavigationFeedbackResponse.STATUS_FOLLOWING_ROUTE,
    graph_nav_pb2.NavigationFeedbackResponse.STATUS_PREPARING_ROBOT,
    }


    # Poll until the command finishes
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not _running:
            logger.info("Stop requested - aborting navigation.")
            return False
        feedback = graph_nav_client.navigation_feedback(nav_to_cmd_id)
        status   = feedback.status

        if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            return True

        if status not in STILL_NAVIGATING:
            status_name = graph_nav_pb2.NavigationFeedbackResponse.Status.Name(status)
            logger.warning("Navigation to %s ended with status: %s", waypoint_id, status_name)
            return False

        time.sleep(0.25)

    logger.warning("Navigation to %s timed out after %.0f s.", waypoint_id, timeout_sec)
    return False



# ======================== GET IMAGE =========================
def capture_at_waypoint(
    robot,
    waypoint: CaptureWaypoint,
    frame_id: int,
    options: GetImageOptions,
    image_results: list,
    colmap_writer: ColmapWriter,
    dry_run: bool,
) -> None:
    """Stop and capture images at the current waypoint."""
    if dry_run:
        logger.info(f"[DRY RUN] Skipping capture at {waypoint.waypoint_id} ({waypoint.label})")
        return

    logger.info(f"Capturing at waypoint {waypoint.waypoint_id} ({waypoint.label}) ...")
    try:
        get_image(
            robot,
            options,
            f"{frame_id:05d}",
            image_results,
            colmap_writer,
        )
        logger.info(
            " Frame %05d captured (%d cameras).",
            frame_id,
            len(options.image_sources or []),
        )
    except Exception as exc:
        logger.warning(f"Capture failed at waypoint {waypoint.waypoint_id}: {exc}")



#============================ Main ===================================

def run_route(args: argparse.Namespace) -> None:
    route_dir = Path(args.route_dir)
    output    = Path(args.output)

    route = RouteDefinition.load(route_dir)
    logger.info(f"\n{route.summary()}\n")

    if not route.capture_waypoints:
        logger.error("No capture waypoints defined in route.json. Aborting.")
        return

    if not route.seed_waypoint_id:
        logger.error("seed_waypoint_id is not set in route.json. Aborting.")
        return

    sdk   = bosdyn.client.create_standard_sdk("spot_route_runner")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    logger.info(f"Connected to robot at {args.hostname}")

    capture_options: GetImageOptions = GetImageOptions(
        output_path=args.output,
        image_sources= [
            "back_depth_in_visual_frame",
            "back_depth",
            "back_fisheye_image",
            "frontleft_depth",
            "frontleft_depth_in_visual_frame",
            "frontleft_fisheye_image",
            "frontright_depth",
            "frontright_depth_in_visual_frame",
            "frontright_fisheye_image",
            "left_depth",
            "left_depth_in_visual_frame",
            "left_fisheye_image",
            "right_depth",
            "right_depth_in_visual_frame",
            "right_fisheye_image",
        ],
        auto_rotate=True,
        save=True,
        show=False,
    )

    sparse_dir    = output / "sparse" / "0"
    colmap_writer = ColmapWriter(sparse_dir)
    image_results: list = []

    lease_client = robot.ensure_client(LeaseClient.default_service_name)    # https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/lease.html
    lease_client.take()
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)

        robot.power_on(timeout_sec=20)
        blocking_stand(robot.ensure_client(RobotCommandClient.default_service_name))
        logger.info("Robot standing and ready.")

        upload_map(graph_nav_client, route_dir)

        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

        if not localise_robot(graph_nav_client, robot_state_client, route.seed_waypoint_id):
            logger.error("Could not localise robot. Check that it is near the start waypoint.")
        else:
            total        = len(route.capture_waypoints)
            frame_id     = 0
            failed_navs  = 0
            MAX_FAILED   = 3   # abort if navigation fails this many times in a row

            logger.info(f"Starting route: {total} capture waypoints.")

            for idx, waypoint in enumerate(route.capture_waypoints):
                if not _running:
                    logger.info("Stop requested - exiting route early.")
                    break

                logger.info(
                    f"===== Waypoint {idx + 1} / {total}  [{waypoint.label}]  id={waypoint.waypoint_id}",
                )

                # Navigate
                reached = navigate_to_waypoint(
                    graph_nav_client,
                    waypoint.waypoint_id,
                    speed_limit=args.nav_velocity,
                    timeout_sec=args.nav_timeout,
                )

                if not reached:
                    failed_navs += 1
                    logger.warning(
                        f"Failed to reach waypoint {waypoint.waypoint_id} ({failed_navs}/{MAX_FAILED} consecutive failures)."
                    )
                    if failed_navs >= MAX_FAILED:
                        logger.error("Too many navigation failures - aborting route.")
                        break
                    continue   # skip capture at this waypoint, try the next

                failed_navs = 0   # reset on success

                # Brief settle pause for reduce/no motion blur 
                time.sleep(args.settle_time)

                # Capture
                frame_id += 1
                capture_at_waypoint(
                    robot, waypoint, frame_id,
                    capture_options, image_results, colmap_writer,
                    dry_run=args.dry_run,
                )


        # ------------- Done with the route ----------------
        logger.info(
            f"\nRoute complete. Frames captured: {frame_id}  |  Images saved: {len(image_results)}",
        )
        logger.info(f"Output: {output.resolve()}")

        # Return the robot to the starting waypoint if requested
        if args.return_to_start and _running:
            logger.info(f"Navigating back to start waypoint ({route.seed_waypoint_id}) ...")
            navigate_to_waypoint(
                graph_nav_client,
                route.seed_waypoint_id,
                speed_limit=args.nav_velocity,
            )



# ======================== Main entry point =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a pre-recorded Spot route and capture images for COLMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hostname",  required=True, help="Robot IP / hostname.")
    parser.add_argument(
        "--route-dir", required=True,
        help="Directory created by record_route.py, e.g. ./routes/lab_loop",
    )
    parser.add_argument(
        "--output", default="./output",
        help="Root directory for COLMAP-ready captured data.",
    )
    parser.add_argument(
        "--capture-sources", nargs="+", default=None, metavar="SOURCE",
        help="Camera sources to capture. Defaults to all five fisheye cameras.",
    )
    parser.add_argument(
        "--nav-velocity", type=float, default=0.8, metavar="M/S",
        help="Maximum navigation speed in m/s (0 = SDK default).",
    )
    parser.add_argument(
        "--nav-timeout", type=float, default=90.0, metavar="SECONDS",
        help="Seconds to wait for the robot to reach a waypoint before giving up.",
    )
    parser.add_argument(
        "--settle-time", type=float, default=1.5, metavar="SECONDS",
        help="Seconds to wait after the robot stops before capturing (reduces motion blur).",
    )
    parser.add_argument(
        "--return-to-start", action="store_true",
        help="Navigate back to the seed waypoint after completing the route.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Navigate the route but do not save any images (for testing).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_route(args)