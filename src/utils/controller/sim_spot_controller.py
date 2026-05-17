
from typing import Callable, List, Tuple



class SimSpotController:

    def __init__(self):
        self.has_lease = False


    # ESTOP

    def estop(self):
        print("Estop")

    def release(self):
        print("Release")

    # GET_IMAGE

    def get_image(self, save = True):
        print("Capturing image from robot's camera...")


    # NAVIGATE ROUTE

    def record_route(self):
        print("Recording route with timestamps and sensor data...")

    def upload_route(self):
        print("Upload a route that the robot should traverse")

    def execute_route(self):
        print("Executing predefined route...")


    # MANUAL RUN

    def manual_run(self):
        print("Manual Controll the robot while he tries to take the Pics...")

    # AUTO RUN

    def auto_run(self) -> Tuple[Callable, Callable, Callable, Callable]:
        print("Starting autonomous 3D environment generation process...")

        return (self._start_navigation, self._stop_navigation, self._intercept_manual_control, self._get_graph)

    def _start_navigation(self):
        print("Starting autonomous navigation...")

    def _stop_navigation(self):
        print("Stopping autonomous navigation...")

    def _intercept_manual_control(self):
        print("Intercepting manual control...")

    def _get_graph(self):
        print("Getting Point Cloud and route graph...")


    # Lease

    def take_lease(self):
        self.has_lease = True
        print("Take lease")

    def release_lease(self):
        self.has_lease = False
        print("Release lease")
    
