from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout

from widgets.sidebar import Sidebar
from widgets.topbar import TopBar
from widgets.content_area import ContentArea
from widgets.bottombar import BottomBar


class MainWindow(QMainWindow):
    
    def __init__(self, controller):
        super().__init__()

        self.setWindowTitle("Autonomous 3D Environment Generation")
        self.resize(1400, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Horizontal Layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left Sidebar
        self.sidebar = Sidebar(controller)

        # Right Side
        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.topbar = TopBar(controller)
        self.content = ContentArea(controller)
        self.bottombar = BottomBar(controller)

        right_layout.addWidget(self.topbar, 1)
        right_layout.addWidget(self.content, 8)
        right_layout.addWidget(self.bottombar, 1)

        right_container.setLayout(right_layout)

        # Add to Main Layout
        main_layout.addWidget(self.sidebar, 2)
        main_layout.addWidget(right_container, 8)

        central_widget.setLayout(main_layout)