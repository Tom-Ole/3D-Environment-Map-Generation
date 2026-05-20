import logging
from pathlib import Path
from typing import Callable, List, Tuple
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

logger = logging.getLogger(__name__)


# util function
def create_check_path(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception(f"Failed to create directory: {path}")
        raise



class SpotController(QObject):

    # Errror 
    error_signal = pyqtSignal(str)

    def __init__(self, robot: Robot, output_path = "./output"):
        super().__init__()
        self.output_path = Path(output_path) / datetime.now().strftime("%Y%m%d_%H_%M%S")
        create_check_path(self.output_path)

        self.robot = robot
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
            except:
                self.estop_endpoint.force_simple_setup()
        else:
            print("No active endpoints for estop register. Force simple setup")
            self.estop_endpoint.force_simple_setup()

        self.estop_keep_alive = EstopKeepAlive(self.estop_endpoint)

        # Lease
        self.has_lease = False

        # Paths
        self.image_output_path = self.output_path / "images"
    
        # Record route
        self._recording_interface = None
        self._record_worker = None

    def _on_close():
        # TODO: handle right closure of ESTOP (remove stop)
        # TODO: handle handback of lease etc. to prev. owner
        # TODO: record graph visulizer needs to be cleaned / or all vtk instances
        # TODO: remvove empty self.output_paths
        pass

    # ESTOP

    def _setup_estop(self):
        pass

    def estop(self):
        self.estop_keep_alive.stop()
        self.is_estop = True

    def release(self):
        self.estop_keep_alive.allow()
        self.is_estop = False

    # GET_IMAGE

    def get_image(self, save = True):
        """ Capture an image from the robot's cameras and save them to a specified location for later processing and 3D reconstruction """
        create_check_path(self.image_output_path)
        
        print("Capturing image from robot's camera...")



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

    def execute_route(self, path: str) -> Callable:
        """Execute a predefined route for the robot to follow"""

        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True) as lease:
                interface = GraphNavInterface(
                                            self.robot, 
                                            path, 
                                            self.command_client, 
                                            self.robot_state_client, 
                                            self.graph_nav_client, 
                                            self.power_client,
                                            lease
                                            )
                interface.run()
                return interface.clear_graph
        except ResourceAlreadyClaimedError:
            print("The robot\'s lease is currently in use. Check for a tablet connection or try again in a few seconds.")
            raise

      


    # MANUAL RUN

    def manual_run(self):
        """Allows to run the Roboter manually while taking the pictures"""
        print("Manual Controll the robot while he tries to take the Pics...")

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

    def take_lease(self):
        """Take the lease if its available"""
        print("Take lease")

    def release_lease(self):
        """Release the lease to other can take it"""
        print("Release lease")
    