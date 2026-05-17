import queue
from datetime import datetime
import threading


from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QMessageBox
)


class Sidebar(QWidget):


    def __init__(self, controller):
        super().__init__()

        self.controller = controller



        layout = QVBoxLayout()
        layout.setSpacing(10)

        # ESTOP button
        self.estop_btn = QPushButton("ESTOP")
        self.estop_btn.clicked.connect(self._estop)
        self.estop_btn.setDisabled(False)
        
        # Release button
        self.release_btn = QPushButton("Release")
        self.release_btn.clicked.connect(self._release)
        self.release_btn.setDisabled(True)

        # Stats Area

        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setText(
            "Some stats \n" 
            "connection infos or so \n"
            "maybe some logs \n"
            "or empty \n" 
        )

        # Bottom Button
        self.lease_btn = QPushButton("Take Lease") # make it dynamic when Lease is taken (should switch to "Return Lease")
        self.lease_btn.clicked.connect(self.lease)

        layout.addWidget(self.estop_btn)
        layout.addWidget(self.release_btn)
        layout.addWidget(self.stats_box)
        layout.addWidget(self.lease_btn)

        self.setLayout(layout)


        # QSS
        self.estop_btn.setObjectName("estopButton")
        self.release_btn.setObjectName("releaseButton")
        self.lease_btn.setObjectName("leaseButton")
    

    def _estop(self):
        self.controller.estop()
        self.release_btn.setDisabled(False)
        self.estop_btn.setDisabled(True)

    def _release(self):
        self.controller.release()
        self.release_btn.setDisabled(True)
        self.estop_btn.setDisabled(False)

    def lease(self):
        if not self.controller.has_lease:   # or whatever your flag is
            self.controller.take_lease()
            self.lease_btn.setText("Return Lease")
        else:
            self.controller.release_lease()
            self.lease_btn.setText("Take Lease")

