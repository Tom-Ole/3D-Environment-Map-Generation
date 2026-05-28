import logging
import shutil
from pathlib import Path
from typing import Callable, Tuple
from datetime import datetime


from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.image import ImageClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
import bosdyn.client.util
from bosdyn.client.robot import PowerClient, Robot

from utils.route.execute_route import GraphNavInterface

from PyQt5.QtCore import QObject, pyqtSignal

from utils.route.record_route import RecordingInterface
from utils.worker.record_worker import RecordWorker
from utils.worker.manual_walk_worker import ManualWalkWorker
from utils.controller.robot_status import RobotStatusSnapshot, format_status_text
from utils.controller.errors import report_error as _emit_error

from utils.preprocess.mask import Preprocessor

from bosdyn.api import robot_state_pb2

logger = logging.getLogger(__name__)

_MOTOR_POWER_LABELS = {
    robot_state_pb2.PowerState.STATE_UNKNOWN: "Unknown",
    robot_state_pb2.PowerState.STATE_ON: "Motors on",
    robot_state_pb2.PowerState.STATE_OFF: "Motors off",
    robot_state_pb2.PowerState.STATE_POWERING_ON: "Powering on",
    robot_state_pb2.PowerState.STATE_POWERING_OFF: "Powering off",
    robot_state_pb2.PowerState.STATE_ERROR: "Power error",
}


# util function
def create_check_path(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception(f"Failed to create directory: {path}")
        raise


def _is_directory_empty(path: Path) -> bool:
    if not path.is_dir():
        return True
    for child in path.iterdir():
        if child.is_file():
            return False
        if child.is_dir() and not _is_directory_empty(child):
            return False
    return True


def remove_path_if_empty(path: Path) -> None:
    if path.exists() and _is_directory_empty(path):
        shutil.rmtree(path)



class SpotController(QObject):

    # Errror 
    error_signal = pyqtSignal(str)


    def __init__(self, robot: Robot, output_path="./output", hostname: str = ""):
        super().__init__()
        self.output_path = Path(output_path) / datetime.now().strftime("%Y%m%d_%H_%M%S")
        create_check_path(self.output_path)

        self.robot = robot
        self.hostname = hostname or getattr(robot, "address", "Spot")
        bosdyn.client.util.authenticate(robot)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync()

        # Clients
        self.estop_client = robot.ensure_client(EstopClient.default_service_name)
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self.graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
        self.power_client = robot.ensure_client(PowerClient.default_service_name)
        self.recording_client = robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
        self.map_processing_client = robot.ensure_client(MapProcessingServiceClient.default_service_name)
        self.lease_client = robot.ensure_client(LeaseClient.default_service_name)

        
        # Estop
        self.is_estop = False
        self.estop_endpoint = EstopEndpoint(self.estop_client,"auto_3D",10.0)

        active_config = self.estop_client.get_config()

        if active_config.endpoints:
            try:
                self.estop_endpoint.register(active_config.unique_id)
            except Exception as e:
                self.estop_endpoint.force_simple_setup()
                self.report_error(f"ESTOP registration failed, using fallback: {e}", e)
        else:
            logger.warning("No active endpoints for estop register; forcing simple setup")
            self.estop_endpoint.force_simple_setup()

        self.estop_keep_alive = EstopKeepAlive(self.estop_endpoint)

        # Lease (acquired automatically by operations such as execute_route)
        self.has_lease = False
        self._lease_keep_alive = None
        self.is_executing_route = False

        # Paths
        self.image_output_path = self.output_path / "images"
    
        # Record route
        self._recording_interface = None
        self._record_worker = None

        # Manual walk
        self._manual_walk_worker = None

        # Preprocessor
        self.preprocessor = Preprocessor()


    def cleanup(self):
        if self.is_estop:
            self.release()
        if self.has_lease:
            self.release_lease()
        try:
            self.estop_keep_alive.shutdown()
        except Exception as e:
            logger.warning("Failed to shutdown estop keep-alive: %s", e)
        remove_path_if_empty(Path(self.output_path))

    def report_error(self, message: str, exc: BaseException | None = None) -> None:
        _emit_error(self.error_signal, message, exc)

    # ESTOP

    def _setup_estop(self):
        pass

    def estop(self):
        self.estop_keep_alive.stop()
        self.is_estop = True

    def release(self):
        self.estop_keep_alive.allow()
        self.is_estop = False

    def _is_recording(self) -> bool:
        worker = self._record_worker
        return worker is not None and worker.isRunning()

    def get_status_snapshot(self) -> RobotStatusSnapshot:
        snapshot = RobotStatusSnapshot(
            hostname=self.hostname,
            estop_active=self.is_estop,
            lease_held=self.has_lease,
            recording=self._is_recording(),
            session_path=str(self.output_path),
        )
        lines = [
            f"Session: {self.output_path.name}",
            f"ESTOP: {'Active' if self.is_estop else 'Released'}",
            f"Lease: {'Held' if self.has_lease else 'Not held'}",
        ]
        if self._is_recording():
            lines.append("Recording: In progress")
        if self.is_executing_route:
            lines.append("Route: Executing")

        try:
            state = self.robot_state_client.get_robot_state()
            snapshot.connected = True

            charge = state.battery_state.charge_percentage
            if charge and charge.value > 0:
                pct = charge.value
                snapshot.battery_percent = pct * 100 if pct <= 1.0 else pct

            motor_state = state.power_state.motor_power_state
            snapshot.motor_power = _MOTOR_POWER_LABELS.get(motor_state, "Unknown")
            lines.append(f"Motors: {snapshot.motor_power}")

            faults = state.behavior_fault_state.faults
            if faults:
                lines.append("Faults:")
                for fault in faults[:5]:
                    name = fault.name or str(fault.behavior_fault_id)
                    lines.append(f"  • {name}")
                if len(faults) > 5:
                    lines.append(f"  … +{len(faults) - 5} more")
        except Exception as e:
            snapshot.connected = False
            lines.append(f"Robot state: unavailable ({e})")
            self.report_error(f"Failed to fetch robot state: {e}", e)

        snapshot.lines = lines
        return snapshot

    def format_status_text(self) -> str:
        return format_status_text(self.get_status_snapshot())

    # GET_IMAGE

    def get_image(self, save = True):
        """ Capture an image from the robot's cameras and save them to a specified location for later processing and 3D reconstruction """
        try:
            create_check_path(self.image_output_path)
            logger.info("Capturing image from robot's camera...")
        except Exception as e:
            self.report_error(f"Failed to capture image: {e}", e)
            raise



    # NAVIGATE ROUTE

    def record_route_start(self,download_filepath: str, session_name: str, user_name: str, on_finished: Callable, on_error: Callable):
        
        client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
        session_name=session_name, client_username=user_name, client_id='RecordingClient',
        client_type='Python SDK')
        self._recording_interface = RecordingInterface(self.robot, 
                                                       download_filepath, 
                                                       client_metadata,
                                                       self.recording_client,
                                                       self.graph_nav_client,
                                                       self.map_processing_client,
                                                        )
        self._record_worker = RecordWorker(self._recording_interface.start)
        self._record_worker.error.connect(on_error)
        self._record_worker.finished.connect(on_finished)
        self._record_worker.start()

    def record_route_waypoint(self, on_error: Callable):
        self._record_worker = RecordWorker(self._recording_interface.create_waypoint)
        self._record_worker.error.connect(on_error)
        self._record_worker.start()

    def record_route_stop(self, create_loop, on_finished: Callable, on_error: Callable):
        self._record_worker = RecordWorker(
            self._recording_interface.stop,
            create_loop=create_loop
        )
        self._record_worker.error.connect(on_error)
        self._record_worker.finished.connect(on_finished)
        self._record_worker.start()

    def execute_route(self, path: str, capture_interval_m: float = 0.1) -> Callable:
        """Execute a predefined route for the robot to follow"""

        self.is_executing_route = True
        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True) as lease_keepalive:
                self._lease_keep_alive = lease_keepalive
                self.has_lease = True
                interface = GraphNavInterface(
                                            self.robot, 
                                            path, 
                                            self.command_client, 
                                            self.robot_state_client, 
                                            self.graph_nav_client, 
                                            self.power_client,
                                            lease_keepalive,
                                            capture_interval_m=capture_interval_m,
                                            )
                interface.run()
                return interface.clear_graph
        except ResourceAlreadyClaimedError as e:
            msg = (
                "The robot's lease is currently in use. Check for a tablet "
                "connection or try again in a few seconds."
            )
            self.report_error(msg, e)
            raise
        finally:
            self._lease_keep_alive = None
            self.has_lease = False
            self.is_executing_route = False

      


    # MANUAL RUN

    def start_manual_run(self, distance_interval_m: float = 1.0, on_finished: Callable = None, on_error: Callable = None):
        """Start manual walk with distance-based image capture."""
        if self._manual_walk_worker is not None and self._manual_walk_worker.isRunning():
            self.report_error("Manual walk is already running.")
            return

        self._manual_walk_worker = ManualWalkWorker(self, distance_interval_m)
        if on_finished:
            self._manual_walk_worker.finished.connect(on_finished)
        if on_error:
            self._manual_walk_worker.error.connect(on_error)
        self._manual_walk_worker.start()

    def stop_manual_run(self, on_finished: Callable = None):
        """Stop manual walk."""
        if self._manual_walk_worker is None or not self._manual_walk_worker.isRunning():
            return

        self._manual_walk_worker.stop()
        if on_finished:
            self._manual_walk_worker.finished.connect(on_finished)

    def manual_capture(self):
        """Manually trigger image capture."""
        try:
            self.get_image(save=True)
        except Exception as e:
            self.report_error(f"Failed to capture image: {e}", e)
            raise

    # AUTO RUN

    def auto_run(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """
        Start the autonomous 3D environment generation process
        It will involve:
        1. Autonomous navigation to explore the environment
        2. Capturing images and sensor data while navigating
        3. Optional manual intervention to guide the robot to specific areas of interest \\
        (if for example the robot is stuck or needs to be guided to a specific area for better data capture) 
        --------
        Returns Functions for:
        1. Start autonomous navigation
        2. Stop autonomous navigation
        3. Intercept manual control (if needed)
        3. Get a Point Cloud of the environment, with the route the robot has already taken so far
        """
        print("Starting autonomous 3D environment generation process...")

        # check if everything is setup
        # for example check if the robot is connected, estop is released, etc.

        return (self._start_navigation, self._stop_navigation, self._intercept_manual_control, self._get_graph)


    def _start_navigation(self):
        """ Start autonomous navigation to explore the environment """
        print("Starting autonomous navigation...")

    def _stop_navigation(self):
        """ Stop autonomous navigation """
        print("Stopping autonomous navigation...")

    def _intercept_manual_control(self):
        """ Intercept manual control for specific interventions """
        print("Intercepting manual control...")

    def _get_graph(self):
        """ Get a Point Cloud of the environment, with the route the robot has already taken so far """
        print("Getting Point Cloud and route graph...")
    

    # Lease

    def release_lease(self):
        """Return the lease so other clients can control the robot."""
        if self._lease_keep_alive is not None:
            try:
                self._lease_keep_alive.return_lease()
            except Exception as e:
                logger.warning("Failed to return lease via keep-alive: %s", e)
                self.report_error(f"Failed to return lease via keep-alive: {e}", e)
            self._lease_keep_alive = None
        else:
            try:
                self.lease_client.return_lease()
            except Exception as e:
                logger.warning("Failed to return lease: %s", e)
                self.report_error(f"Failed to return lease: {e}", e)
        self.has_lease = False
    
    # Preprocessor

    def create_masks(self, input_path: str, output_path: str, classes: list[int] = [0]):
        """Create masks for the captured images using the Preprocessor."""
        self.preprocessor.create_masks(input_path, output_path, classes)

    def create_masks_recursive(self, input_path: str, output_path: str, classes: list[int] = [0]):
        self.preprocessor.create_masks_recursive(input_path, output_path, classes)