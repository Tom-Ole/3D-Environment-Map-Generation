from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

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



def upload_map(graph_nav_client: GraphNavClient, route_dir: Path): # -> map_pb2.Graph
    """
    Upload the graph topology and all waypoint snapshots from *route_dir* to
    the robot.  Returns the deserialized Graph proto.
    """
    logger.info(f"Uploading map from {route_dir} ...")

    graph_path = route_dir / "graph"
    with open(graph_path, "rb") as fh:
        graph = map_pb2.Graph()
        graph.ParseFromString(fh.read())

    graph_nav_client.upload_graph(graph=graph)
    logger.info(f"Graph uploaded  ({len(graph.waypoints)} waypoints, {len(graph.edges)} edges)")


    snapshots_dir = route_dir / "waypoint_snapshots"
    uploaded = 0
    for waypoint in graph.waypoints:
        snap_path = snapshots_dir / waypoint.snapshot_id
        if not snap_path.exists():
            logger.warning(f"Snapshot missing: {snap_path}")
            continue
        with open(snap_path, "rb") as fh:
            snapshot = map_pb2.WaypointSnapshot()
            snapshot.ParseFromString(fh.read())
        graph_nav_client.upload_waypoint_snapshot(snapshot)
        uploaded += 1

    logger.info(f"Uploaded {uploaded} / {len(graph.waypoints)} waypoint snapshots.")
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
            max_distance=1.0,
            max_yaw=1.57,
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


def validate_route(route: RouteDefinition) -> bool:
    seen_ids: dict[str, str] = {}  # waypoint_id -> label
    valid = True
    for wp in route.capture_waypoints:
        if wp.waypoint_id in seen_ids:
            logger.warning(
                f"Duplicate waypoint ID '{wp.waypoint_id}' used by both "
                f"'{seen_ids[wp.waypoint_id]}' and '{wp.label}'. "
                f"The robot will not move between these stops."
            )
            valid = False
        else:
            seen_ids[wp.waypoint_id] = wp.label
    return valid

# https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/graph_nav/graph_nav.proto
_STATUS = graph_nav_pb2.NavigationFeedbackResponse

STILL_NAVIGATING = {
    _STATUS.STATUS_FOLLOWING_ROUTE,
}

# These are worth retrying - transient physical/timing issues
RETRYABLE = {
    _STATUS.STATUS_STUCK,
    _STATUS.STATUS_LOST,
    _STATUS.STATUS_COMMAND_TIMED_OUT,
}

# These indicate a fundamental problem - retrying will not help
TERMINAL = {
    _STATUS.STATUS_NO_ROUTE,
    _STATUS.STATUS_NO_LOCALIZATION,
    _STATUS.STATUS_NOT_LOCALIZED_TO_ROUTE,
    _STATUS.STATUS_ROBOT_IMPAIRED,
    _STATUS.STATUS_CONSTRAINT_FAULT,
    _STATUS.STATUS_COMMAND_OVERRIDDEN,
    _STATUS.STATUS_LEASE_ERROR,
    _STATUS.STATUS_AREA_CALLBACK_ERROR,
    _STATUS.STATUS_UNKNOWN,
}


def navigate_to_waypoint(
    graph_nav_client: GraphNavClient,
    waypoint_id: str,
    speed_limit: float = 0.0,
    timeout_sec: float = 60.0,
    max_retries: int = 2,
) -> bool:
    nav_params = None
    if speed_limit > 0:
        nav_params = graph_nav_pb2.TravelParams(
            max_distance=0.0,
            velocity_limit=geometry_pb2.SE2VelocityLimit(
                max_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=speed_limit, y=0),
                    angular=0,
                )
            ),
        )

    for attempt in range(1, max_retries + 2):
        try:
            nav_to_cmd_id = graph_nav_client.navigate_to(
                waypoint_id,
                travel_params=nav_params,
                cmd_duration=100,
            )
        except Exception as exc:
            logger.error(f"navigate_to({waypoint_id}) call failed: {exc}")
            return False

        retry_reason: str | None = None

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if not _running:
                logger.info("Stop requested - aborting navigation.")
                return False

            feedback = graph_nav_client.navigation_feedback(nav_to_cmd_id)
            status   = feedback.status

            if status == _STATUS.STATUS_REACHED_GOAL:
                return True

            if status in STILL_NAVIGATING:
                time.sleep(0.25)
                continue

            status_name = _STATUS.Status.Name(status)

            if status in TERMINAL:
                logger.error(f"Terminal navigation failure for {waypoint_id}: {status_name}")
                if status == _STATUS.STATUS_ROBOT_IMPAIRED:
                    logger.error(f"  Impaired status: {feedback.impaired_status}")
                return False

            if status in RETRYABLE:
                extra = ""
                if status == _STATUS.STATUS_STUCK and feedback.HasField("stuck_reason"):
                    extra = f"- {feedback.stuck_reason}"
                logger.warning(
                    f"{status_name}{extra} navigating to {waypoint_id} "
                    f"(attempt {attempt}/{max_retries + 1})"
                )
                retry_reason = status_name
                break  # exit poll loop -> retry

            # Catch any future statuses not yet in our sets
            logger.warning(f"Unhandled navigation status {status_name} for {waypoint_id}")
            return False

        else:
            # Poll loop exhausted its deadline
            logger.warning(f"Navigation to {waypoint_id} timed out after {timeout_sec:.0f}s.")
            retry_reason = "TIMEOUT"

        if attempt > max_retries:
            logger.error(
                f"Giving up on {waypoint_id} after {attempt} attempt(s). "
                f"Last reason: {retry_reason}"
            )
            return False

        logger.info(f"Retrying {waypoint_id} in 2 s... (attempt {attempt + 1}/{max_retries + 1})")
        time.sleep(2.0)

    return False

def capture_at_waypoint(
    robot,
    waypoint: CaptureWaypoint,
    frame_id: int,
    options: GetImageOptions,
    image_results: list,
    colmap_writer: ColmapWriter,
    dry_run: bool,
    lease = None
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
            lease=lease
        )
        logger.info(f" Frame {frame_id:05d} captured ({len(options.image_sources or [])} cameras).")
    except Exception as exc:
        logger.warning(f"Capture failed at waypoint {waypoint.waypoint_id}: {exc}")


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
            #"back_depth_in_visual_frame",
            #"back_depth",
            "back_fisheye_image",
            #"frontleft_depth",
            #"frontleft_depth_in_visual_frame",
            "frontleft_fisheye_image",
            #"frontright_depth",
            #"frontright_depth_in_visual_frame",
            "frontright_fisheye_image",
            #"left_depth",
            #"left_depth_in_visual_frame",
            "left_fisheye_image",
            #"right_depth",
            #"right_depth_in_visual_frame",
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
                    f"===== Waypoint {idx + 1} / {total}  [{waypoint.label}]  id={waypoint.waypoint_id} ============",
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
                    lease=lease_client
                )


        # Done with the route
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




# =================================================

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