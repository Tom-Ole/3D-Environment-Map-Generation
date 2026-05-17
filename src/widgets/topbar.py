from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
)

import PyQt5.QtCore as QtCore


class TopBar(QWidget):

    auto_clicked = QtCore.pyqtSignal()
    record_clicked = QtCore.pyqtSignal()
    upload_clicked = QtCore.pyqtSignal()
    manual_clicked = QtCore.pyqtSignal()

    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        layout = QHBoxLayout()
        layout.setSpacing(10)

        self.auto_walk_btn = QPushButton("Auto walk")
        self.rec_route_btn = QPushButton("Rec. Route")
        self.upload_route_btn = QPushButton("Upload Route")
        self.manual_walk_btn = QPushButton("Manual Walk")


        self.auto_walk_btn.clicked.connect(self.auto_clicked.emit)
        self.rec_route_btn.clicked.connect(self.record_clicked.emit)
        self.upload_route_btn.clicked.connect(self.upload_clicked.emit)
        self.manual_walk_btn.clicked.connect(self.manual_clicked.emit)
        

        layout.addWidget(self.auto_walk_btn)
        layout.addWidget(self.rec_route_btn)
        layout.addWidget(self.upload_route_btn)
        layout.addWidget(self.manual_walk_btn)

        self.setLayout(layout)
