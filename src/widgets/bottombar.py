from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt


class BottomBar(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        layout = QHBoxLayout()

        status = QLabel("Status / Console / Logs")
        status.setAlignment(Qt.AlignCenter)

        layout.addWidget(status)

        self.setLayout(layout)