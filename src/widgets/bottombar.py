from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer


class BottomBar(QWidget):

    def __init__(self, controller):
        super().__init__()
        self.setObjectName("bottomBar")

        self.controller = controller

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        self._status = QLabel("Ready")
        self._status.setObjectName("statusLabel")
        self._status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self._error = QLabel("")
        self._error.setObjectName("errorLabel")
        self._error.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._error.setWordWrap(True)

        self._clear_btn = QPushButton("\u2715")
        self._clear_btn.setObjectName("clearErrorBtn")
        self._clear_btn.setToolTip("Clear error message")
        self._clear_btn.clicked.connect(self._clear_error)
        self._clear_btn.setVisible(False)

        layout.addWidget(self._status, stretch=1)
        layout.addWidget(self._error, stretch=2)
        layout.addWidget(self._clear_btn)

        self.setLayout(layout)

        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self._clear_error)

        self.controller.error_signal.connect(self._show_error)

    def set_status(self, message: str):
        self._status.setText(message)

    def _show_error(self, message: str):
        self._error.setText(message)
        self._clear_btn.setVisible(True)
        self._error_timer.stop()
        self._error_timer.start(10_000)

    def _clear_error(self):
        self._error_timer.stop()
        self._error.setText("")
        self._clear_btn.setVisible(False)
