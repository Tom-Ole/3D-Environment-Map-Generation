from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PyQt5.QtCore import Qt


class ContentArea(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        layout = QVBoxLayout()

        text = (
            "Auto Walk: 3D point of the environment that he was near.\n\n"
            "Rec. Route: 3D point of the environment that he was near.\n\n"
            "Upload route: Route visualized\n\n"
            "Manual Walk: Dont know right now"
        )

        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)

        layout.addStretch()
        layout.addWidget(label)
        layout.addStretch()

        self.setLayout(layout)