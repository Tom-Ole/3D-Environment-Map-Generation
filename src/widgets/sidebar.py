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
        self.estop_btn.setMinimumHeight(220)
        self.estop_btn.clicked.connect(self.controller.estop)
        
        # Release button
        self.release_btn = QPushButton("Release")
        self.release_btn.clicked.connect(self.controller.release)
        self.release_btn.setMinimumHeight(120)

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
        self.lease_btn.setMinimumHeight(70)

        layout.addWidget(self.estop_btn)
        layout.addWidget(self.release_btn)
        layout.addWidget(self.stats_box)
        layout.addWidget(self.lease_btn)

        self.setLayout(layout)

    def lease(self):
        self.controller.take_lease()
        self.controller.release_lease()

