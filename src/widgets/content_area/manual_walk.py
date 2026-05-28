from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
)

from widgets.graph_nav_visualizer import GraphNavWidget
from widgets.content_area.helpers import (
    section_label,
    separator,
    control_panel,
    map_page,
)


class ManualWalkPanel(QWidget):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

        panel = control_panel()
        ctrl = QVBoxLayout(panel)
        ctrl.setContentsMargins(12, 12, 12, 12)
        ctrl.setSpacing(8)

        ctrl.addWidget(section_label("Capture Settings"))
        
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Distance interval (m):"))
        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setRange(0.1, 100.0)
        self._interval_spin.setValue(1.0)
        self._interval_spin.setDecimals(2)
        self._interval_spin.setToolTip(
            "Distance interval in meters at which the robot should take pictures automatically."
        )
        interval_row.addWidget(self._interval_spin)
        ctrl.addLayout(interval_row)
        ctrl.addWidget(separator())

        ctrl.addWidget(section_label("Manual Controls"))
        self._capture_btn = QPushButton("Capture Image")
        self._capture_btn.setEnabled(False)
        ctrl.addWidget(self._capture_btn)
        ctrl.addWidget(separator())

        ctrl.addWidget(section_label("Walk Control"))
        self._start_btn = QPushButton("Start")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        
        ctrl.addWidget(self._start_btn)
        ctrl.addWidget(self._stop_btn)
        ctrl.addStretch()

        self.graph_widget = GraphNavWidget(placeholder_text="")
        self.graph_widget.load_failed.connect(self._on_error)

        self._capture_btn.clicked.connect(self._on_capture)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(map_page(panel, self.graph_widget))

    def _on_error(self, message: str):
        self.controller.report_error(message)

    def _on_capture(self):
        try:
            self.controller.manual_capture()
        except Exception:
            pass

    def _on_start(self):
        self.controller.start_manual_run(
            distance_interval_m=self._interval_spin.value(),
            on_finished=self._on_start_finished,
            on_error=self._on_worker_error,
        )
        self._start_btn.setEnabled(False)
        self._capture_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._interval_spin.setEnabled(False)

    def _on_start_finished(self):
        self._start_btn.setEnabled(True)
        self._capture_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._interval_spin.setEnabled(True)

    def _on_stop(self):
        self.controller.stop_manual_run(on_finished=self._on_start_finished)

    def _on_worker_error(self, exc: Exception):
        self.controller.report_error(str(exc), exc)
        self._on_start_finished()

    def cleanup(self):
        self.graph_widget.cleanup_vtk()
