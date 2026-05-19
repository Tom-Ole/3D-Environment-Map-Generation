from PyQt5.QtCore import QThread, pyqtSignal

class RouteWorker(QThread):
    finished = pyqtSignal(object)   
    error = pyqtSignal(Exception)

    def __init__(self, controller, folder):
        super().__init__()
        self.controller = controller
        self.folder = folder

    def run(self):
        try:
            clear_fn = self.controller.execute_route(self.folder)
            self.finished.emit(clear_fn)
        except Exception as e:
            self.error.emit(e)