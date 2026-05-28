from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from utils.controller.spot_controller import remove_path_if_empty
from utils.controller.robot_status import RobotStatusSnapshot, format_status_text
from utils.controller.errors import report_error as _emit_error
import logging

from utils.preprocess.mask import Preprocessor

logger = logging.getLogger(__name__)


class SimSpotController(QObject):
    error_signal = pyqtSignal(str)

    def __init__(self, output_path: str = "./output", hostname: str = ""):
        super().__init__()
        self.has_lease = False
        self.is_estop = False
        self.robot = None
        self.graph_nav_client = None
        self._recording = False
        self.hostname = hostname or "(simulation)"
        self.is_executing_route = False

        self.output_path = Path(output_path) / datetime.now().strftime("%Y%m%d_%H_%M%S")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.image_output_path = self.output_path / "images"

        self.preprocessor = Preprocessor()

    def cleanup(self):
        if self.is_estop:
            self.release()
        if self.has_lease:
            self.release_lease()
        remove_path_if_empty(self.output_path)

    def report_error(self, message: str, exc: BaseException | None = None) -> None:
        _emit_error(self.error_signal, message, exc)

    # ESTOP

    def estop(self):
        self.is_estop = True
        print("Estop")

    def release(self):
        self.is_estop = False
        print("Release")

    def get_status_snapshot(self) -> RobotStatusSnapshot:
        lines = [
            "Mode: Simulation",
            f"Session: {self.output_path.name}",
            f"ESTOP: {'Active' if self.is_estop else 'Released'}",
            f"Lease: {'Held' if self.has_lease else 'Not held'}",
            f"Recording: {'In progress' if self._recording else 'Idle'}",
            f"Route: {'Executing' if self.is_executing_route else 'Idle'}",
            "Motors: N/A (sim)",
        ]
        return RobotStatusSnapshot(
            hostname=self.hostname,
            connected=True,
            battery_percent=None,
            motor_power="Simulation",
            estop_active=self.is_estop,
            lease_held=self.has_lease,
            recording=self._recording,
            session_path=str(self.output_path),
            lines=lines,
        )

    def format_status_text(self) -> str:
        return format_status_text(self.get_status_snapshot())

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

    def execute_route(self, path: str, capture_interval_m: float = 0.1) -> Callable:
        logger.info(
            "[Sim] Executing route [%s] (interval=%.2fm)", path, capture_interval_m
        )
        self.is_executing_route = True
        self.has_lease = True

        def clear_fn():
            logger.info("[Sim] Cleared graph for route [%s]", path)
            self.has_lease = False
            self.is_executing_route = False

        return clear_fn

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

    def release_lease(self):
        self.has_lease = False
        print("[Sim] Lease released")

    # PREPROCESSING
    def create_masks(self, input_path: str, output_path: str, classes: list[int] = [0]):
        """Create masks for the captured images using the Preprocessor."""
        self.preprocessor.create_masks(input_path, output_path, classes)

    def create_masks_recursive(self, input_path: str, output_path: str, classes: list[int] = [0]):
        self.preprocessor.create_masks_recursive(input_path, output_path, classes)