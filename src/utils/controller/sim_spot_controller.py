from typing import Callable, Tuple
from PyQt5.QtCore import QObject, pyqtSignal


class SimSpotController(QObject):
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.has_lease = False
        self.robot = None
        self.graph_nav_client = None
        self._recording = False

    # ESTOP

    def estop(self):
        print("Estop")

    def release(self):
        print("Release")

    # GET_IMAGE

    def get_image(self, save=True):
        print("Capturing image from robot's camera...")

    # RECORD ROUTE

    def record_route_start(self, download_filepath: str, session_name: str, user_name: str,
                           on_finished: Callable, on_error: Callable):
        print(f"[Sim] Start recording → {download_filepath} | session={session_name} | user={user_name}")
        self._recording = True
        on_finished()

    def record_route_waypoint(self, on_error: Callable):
        if not self._recording:
            on_error(RuntimeError("Not currently recording."))
            return
        print("[Sim] Create waypoint")

    def record_route_stop(self, create_loop: bool, on_finished: Callable, on_error: Callable):
        if not self._recording:
            on_error(RuntimeError("Not currently recording."))
            return
        print(f"[Sim] Stop recording | create_loop={create_loop}")
        self._recording = False
        on_finished()

    # EXECUTE ROUTE

    def execute_route(self, path: str) -> Callable:
        print(f"[Sim] Executing predefined route [{path}]...")
        return lambda: print(f"[Sim] Cleared graph for route [{path}]")

    # MANUAL RUN

    def manual_run(self):
        print("[Sim] Manual control...")

    # AUTO RUN

    def auto_run(self) -> Tuple[Callable, Callable, Callable, Callable]:
        print("[Sim] Starting autonomous process...")
        return (self._start_navigation, self._stop_navigation,
                self._intercept_manual_control, self._get_graph)

    def _start_navigation(self):
        print("[Sim] Starting autonomous navigation...")

    def _stop_navigation(self):
        print("[Sim] Stopping autonomous navigation...")

    def _intercept_manual_control(self):
        print("[Sim] Intercepting manual control...")

    def _get_graph(self):
        print("[Sim] Getting Point Cloud and route graph...")

    # LEASE

    def take_lease(self):
        self.has_lease = True
        print("[Sim] Lease taken")

    def release_lease(self):
        self.has_lease = False
        print("[Sim] Lease released")