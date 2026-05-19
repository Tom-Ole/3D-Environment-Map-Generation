from PyQt5.QtCore import QThread, pyqtSignal

class RecordWorker(QThread):
    error = pyqtSignal(Exception)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self._fn(*self._args, **self._kwargs)
        except Exception as e:
            self.error.emit(e)