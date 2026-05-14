from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
)


class TopBar(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        layout = QHBoxLayout()
        layout.setSpacing(10)

        self.auto_walk_btn = QPushButton("Auto walk")
        self.auto_walk_btn.clicked.connect(self.auto_walk)
        self.rec_route_btn = QPushButton("Rec. Route")
        self.rec_route_btn.clicked.connect(self.controller.record_route)
        self.upload_route_btn = QPushButton("Upload Route")
        self.upload_route_btn.clicked.connect(self.controller.upload_route)
        self.manual_walk_btn = QPushButton("Manual Walk")
        self.manual_walk_btn.clicked.connect(self.controller.manual_run)
        

        layout.addWidget(self.auto_walk_btn)
        layout.addWidget(self.rec_route_btn)
        layout.addWidget(self.upload_route_btn)
        layout.addWidget(self.manual_walk_btn)

        self.setLayout(layout)

    def auto_walk(self):
        start_fn, stop_fn, intercept_fn, get_graph_fn = self.controller.auto_run()
        start_fn()
        stop_fn()
        intercept_fn()
        get_graph_fn()
