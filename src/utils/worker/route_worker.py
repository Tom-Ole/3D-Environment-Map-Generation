from PyQt5.QtCore import QThread, pyqtSignal


class RouteWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)

    def __init__(self, controller, folder, capture_interval_m: float = 0.1):
        super().__init__()
        self.controller = controller
        self.folder = folder
        self.capture_interval_m = capture_interval_m

    def run(self):
        try:
            clear_fn = self.controller.execute_route(
                self.folder, capture_interval_m=self.capture_interval_m
            )
            self.finished.emit(clear_fn)
        except Exception as e:
            self.error.emit(e)
