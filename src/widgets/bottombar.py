from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt


class BottomBar(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        layout = QHBoxLayout()

        self._status = QLabel(" ")
        self._status.setAlignment(Qt.AlignCenter)

        layout.addWidget(self._status)

        self.setLayout(layout)

        self.controller.error_signal.connect(self._show_error)

    def _show_error(self, message: str):
        self._status.setText(f"Error: {message}")
        self._status.setStyleSheet("color: red;")