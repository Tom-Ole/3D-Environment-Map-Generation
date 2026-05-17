import logging
from pathlib import Path
from typing import Callable, List, Tuple
from datetime import datetime


from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.image import ImageClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
import bosdyn.client.util
from bosdyn.api import estop_pb2 as estop_protos
from bosdyn.client.robot import Robot

logger = logging.getLogger(__name__)


# util function
def create_check_path(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception(f"Failed to create directory: {path}")
        raise



class SpotController:

    def __init__(self, robot: Robot, output_path = "./output"):

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
        
        # Estop
        self.is_estop = False
        self.estop_endpoint = EstopEndpoint(self.estop_client,"GUI",5.0)

        self.estop_endpoint.force_simple_setup()
        self.estop_keep_alive = EstopKeepAlive(self.estop_endpoint)

        # Lease
        self.has_lease = False

        # Paths
        self.image_output_path = self.output_path / "images"
    

    # ESTOP

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

    def record_route(self):
        """ Record the route taken by the robot during navigation, including timestamps and sensor data for each point in the route """
        print("Recording route with timestamps and sensor data...")

    def upload_route(self):
        """Upload a route that the robot should traverse"""
        print("Upload a route that the robot should traverse")

    def execute_route(self):
        """ Execute a predefined route for the robot to follow """
        print("Executing predefined route...")

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
    