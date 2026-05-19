
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import bosdyn.client
import bosdyn.client.util
from bosdyn.api.graph_nav import recording_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from scipy.spatial.transform import Rotation
import queue
import threading

from old._route.route import RouteDefinition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def download_map(graph_nav_client: GraphNavClient, route_dir: Path) -> None:
    """
    Download the full GraphNav map (graph + all waypoint snapshots) from the
    robot and write it to *route_dir*.
    """
    logger.info("Downloading map from robot …")
    route_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = route_dir / "waypoint_snapshots"
    snapshots_dir.mkdir(exist_ok=True)

    graph = graph_nav_client.download_graph()
    graph_path = route_dir / "graph"
    with open(graph_path, "wb") as fh:
        fh.write(graph.SerializeToString())
    logger.info(f"Graph saved  ->  {graph_path}  ({len(graph.waypoints)} waypoints, {len(graph.edges)} edges)")

    for waypoint in graph.waypoints:
        if not waypoint.snapshot_id:
            continue
        snapshot = graph_nav_client.download_waypoint_snapshot(waypoint.snapshot_id)
        snap_path = snapshots_dir / waypoint.snapshot_id
        with open(snap_path, "wb") as fh:
            fh.write(snapshot.SerializeToString())

    logger.info(f"Waypoint snapshots saved  ->  {snapshots_dir}")



def get_robot_se2(graph_nav_client: GraphNavClient) -> Optional[tuple[float, float, float]]:
    """
    Return (x, y, yaw) of the robot in the graph's seed frame, or None if
    localisation is not available.
    """
    state = graph_nav_client.get_localization_state()
    loc   = state.localization
    if not loc.waypoint_id:
        return None
    x   = loc.waypoint_tform_body.position.x
    y   = loc.waypoint_tform_body.position.y
    q = loc.waypoint_tform_body.rotation
    yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
    return x, y, yaw


def se2_distance(a: tuple, b: tuple) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


class RouteRecorder:
    """
    Manages GraphNav recording and interactive waypoint marking.
    """

    def __init__(
        self,
        robot,
        route_dir: Path,
        description: str,
        auto_capture_distance: float,
    ) -> None:
        self.robot                 = robot
        self.route_dir             = route_dir
        self.auto_capture_distance = auto_capture_distance

        self.graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
        self.recording_client = robot.ensure_client(
            GraphNavRecordingServiceClient.default_service_name
        )
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        self.route = RouteDefinition(
            route_dir=str(route_dir),
            description=description,
            auto_capture_distance=auto_capture_distance,
        )
        self._last_auto_pose: Optional[tuple] = None

    def start_recording(self) -> None:
        """Clear any existing map on the robot and begin recording."""

        try:
            self.recording_client.stop_recording()
        except Exception:
            pass  # not recording

        # Clear the existing map so we start fresh
        self.graph_nav_client.clear_graph()

        env = recording_pb2.RecordingEnvironment()
        env.name_prefix = "spot_route_tom"
        self.recording_client.start_recording(recording_environment=env)
        logger.info("GraphNav recording started.")

    def stop_recording(self) -> str:
        """Stop recording and return the ID of the last waypoint."""
        response = self.recording_client.stop_recording()
        logger.info("GraphNav recording stopped.")
        return response

    def get_current_waypoint_id(self) -> Optional[str]:
        """Return the ID of the waypoint closest to the robot right now."""
        state = self.graph_nav_client.get_localization_state()
        loc   = state.localization
        return loc.waypoint_id if loc.waypoint_id else None

    def mark_capture_waypoint(self, label: str = "", notes: str = "") -> Optional[str]:
        """
        Create a GraphNav anchor at the current position and register it as a
        capture waypoint.  Returns the waypoint ID, or None on failure.
        """
        waypoint_id = self.get_current_waypoint_id()
        if not waypoint_id:
            logger.warning("Cannot mark waypoint - robot is not localised yet.")
            return None

        # Compute distance from previous capture waypoint
        dist = 0.0
        if self.route.capture_waypoints:
            pose = get_robot_se2(self.graph_nav_client)
            if pose and self._last_auto_pose:
                dist = se2_distance(pose, self._last_auto_pose)
            self._last_auto_pose = pose
        else:
            # First waypoint → also use as the seed for localisation on replay
            self.route.seed_waypoint_id = waypoint_id
            self._last_auto_pose = get_robot_se2(self.graph_nav_client)

        self.route.add_capture_waypoint(
            waypoint_id=waypoint_id,
            label=label or f"wp_{len(self.route.capture_waypoints):03d}",
            distance_from_prev=dist,
            notes=notes,
        )
        logger.info(
            f"Marked capture waypoint #{len(self.route.capture_waypoints)}  id={waypoint_id}  label={self.route.capture_waypoints[-1].label!r}  dist_prev={dist:.2f}m"
        )
        return waypoint_id

    def check_auto_capture(self) -> None:
        """
        If auto_capture_distance > 0, automatically mark a waypoint whenever
        the robot has moved at least that far from the last capture point.
        Called periodically from the main loop.
        """
        if self.auto_capture_distance <= 0:
            return
        pose = get_robot_se2(self.graph_nav_client)
        if pose is None:
            return
        if self._last_auto_pose is None:
            self._last_auto_pose = pose
            return
        if se2_distance(pose, self._last_auto_pose) >= self.auto_capture_distance:
            self.mark_capture_waypoint()
            self._last_auto_pose = get_robot_se2(self.graph_nav_client)

    def save(self) -> None:
        """Download the map from the robot and save everything to disk."""
        download_map(self.graph_nav_client, self.route_dir)
        saved_path = self.route.save()
        logger.info(f"Route saved  ->  {saved_path}")
        print("\n" + self.route.summary())

# ===================================================================================
# ===================================================================================


def _prompt_mark(recorder: RouteRecorder) -> None:
    """Ask the user for a label and mark a capture waypoint."""
    label = input("  Label (leave blank for auto): ").strip()
    notes = "" #input("  Notes (optional)           : ").strip()
    wid   = recorder.mark_capture_waypoint(label=label, notes=notes)
    if wid:
        print(f"   Waypoint marked: {wid} \n")
    else:
        print("   Could not mark waypoint.")


def recording_loop(recorder: RouteRecorder) -> None:

    print("\n" + "-" * 60)
    print("RECORDING ACTIVE")
    print("  ENTER           -> mark current position as capture waypoint")
    print("  q + ENTER       -> stop recording and save")
    print("  l + ENTER       -> list marked waypoints so far")
    print("-" * 60 + "\n")

    stop_event = threading.Event()

    def auto_capture_worker():
        while not stop_event.is_set():
            recorder.check_auto_capture()
            time.sleep(0.3)

    worker = threading.Thread(target=auto_capture_worker, daemon=True)
    worker.start()
    
    try:
        while True:
            line = input("Command: ").strip().lower()

            if line == "":
                _prompt_mark(recorder)
            elif line == "q":
                logger.info("Stopping recording …")
                break
            elif line == "l":
                if not recorder.route.capture_waypoints:
                    print("  (no waypoints marked yet)")
                else:
                    for i, wp in enumerate(recorder.route.capture_waypoints):
                        print(f"  [{i:3d}] {wp.label:<20s}  {wp.waypoint_id}")
            else:
                print("  Unknown command. ENTER=mark  q=quit  l=list")
    finally:
        stop_event.set()
        worker.join()
 
        

def record_run(args: argparse.Namespace) -> None:
    route_dir = Path(args.route_dir)

    sdk   = bosdyn.client.create_standard_sdk("spot_route_recorder")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    logger.info(f"Connected to robot at {args.hostname}")


    recorder = RouteRecorder(
        robot=robot,
        route_dir=route_dir,
        description=args.description,
        auto_capture_distance=args.auto_capture_distance,
    )

    recorder.start_recording()

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if recorder.get_current_waypoint_id():
            break
        time.sleep(0.5)
    else:
        logger.error("Robot did not localise within 10s of starting recording.")
        return

    recorder.mark_capture_waypoint(label="start")

    try:
        recording_loop(recorder)
    finally:
        recorder.stop_recording()
        recorder.save()

    logger.info(f"Done. You can now run:  python run_route.py --route-dir {route_dir} --hostname {args.hostname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a Spot GraphNav route for 3D reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hostname",  required=True, help="Robot IP / hostname.")
    parser.add_argument(
        "--route-dir", required=True,
        help="Directory to save the map and route.json, e.g. ./routes/lab_loop",
    )
    parser.add_argument(
        "--description", default="",
        help="Free-text description of the route.",
    )
    parser.add_argument(
        "--auto-capture-distance", type=float, default=0.0, metavar="METRES",
        help=(
            "Automatically mark a capture waypoint every N metres of travel. "
            "Set to 0 to disable (manual-only mode)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    record_run(args)