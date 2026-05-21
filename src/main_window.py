from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent

from widgets.sidebar import Sidebar
from widgets.content_area import ContentArea
from widgets.bottombar import BottomBar


class MainWindow(QMainWindow):

    def __init__(self, controller, hostname: str = "192.168.10.3"):
        super().__init__()

        self.controller = controller

        self.setWindowTitle("Autonomous 3D Environment Generation")
        self.setMinimumSize(1100, 750)
        self.resize(1400, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root = QVBoxLayout(central_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)

        self.sidebar = Sidebar(controller, hostname=hostname)
        self.content = ContentArea(controller)

        body.addWidget(self.sidebar)
        body.addWidget(self.content)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)

        self.bottombar = BottomBar(controller)

        root.addWidget(body, stretch=1)
        root.addWidget(self.bottombar)

    def closeEvent(self, event: QCloseEvent):
        self.sidebar.stop_polling()
        self.content.cleanup()
        if hasattr(self.controller, "cleanup"):
            self.controller.cleanup()
        super().closeEvent(event)
