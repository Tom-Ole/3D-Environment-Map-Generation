from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QDoubleSpinBox,
)
from PyQt5.QtCore import pyqtSignal

from widgets.graph_nav_visualizer import GraphNavWidget
from utils.worker.route_worker import RouteWorker
from widgets.content_area.helpers import (
    section_label,
    muted_label,
    separator,
    control_panel,
    set_path_label,
    map_page,
)


class RunRoutePanel(QWidget):
    route_started = pyqtSignal()
    route_finished = pyqtSignal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.clear_uploaded_route = None
        self._selected_folder: str | None = None
        self._route_worker = None

        panel = control_panel()
        ctrl = QVBoxLayout(panel)
        ctrl.setContentsMargins(12, 12, 12, 12)
        ctrl.setSpacing(8)

        ctrl.addWidget(section_label("GraphNav Map"))
        self._path_label = muted_label("No folder selected")
        browse_btn = QPushButton("Browse\u2026")
        self._load_btn = QPushButton("Load Map")
        self._load_btn.setEnabled(False)
        clear_btn = QPushButton("Clear")

        self._anchoring_chk = QCheckBox("Use anchoring")
        self._wp_text_chk = QCheckBox("Waypoint labels")
        self._wp_text_chk.setChecked(True)
        self._wo_text_chk = QCheckBox("World-object labels")
        self._wo_text_chk.setChecked(True)

        ctrl.addWidget(browse_btn)
        ctrl.addWidget(self._path_label)
        ctrl.addWidget(separator())
        ctrl.addWidget(self._anchoring_chk)
        ctrl.addWidget(self._wp_text_chk)
        ctrl.addWidget(self._wo_text_chk)
        ctrl.addWidget(self._load_btn)
        ctrl.addWidget(clear_btn)
        ctrl.addWidget(separator())

        ctrl.addWidget(section_label("Capture Options"))
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Interval (m):"))
        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setRange(0.0, 100.0)
        self._interval_spin.setValue(0.1)
        self._interval_spin.setDecimals(2)
        self._interval_spin.setToolTip(
            "Minimum distance between captures at waypoints. "
            "Set to 0 to capture at every waypoint."
        )
        interval_row.addWidget(self._interval_spin)
        ctrl.addLayout(interval_row)
        ctrl.addWidget(separator())

        ctrl.addWidget(section_label("Start Capturing"))
        self._start_btn = QPushButton("Start")
        ctrl.addWidget(self._start_btn)
        ctrl.addStretch()

        self.graph_widget = GraphNavWidget()
        self.graph_widget.load_failed.connect(self._on_error)

        browse_btn.clicked.connect(self._on_browse)
        self._load_btn.clicked.connect(self._on_load)
        clear_btn.clicked.connect(self._on_clear)
        self._start_btn.clicked.connect(self._on_start)

        self._apply_default_route_folder()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(map_page(panel, self.graph_widget))

    def _on_error(self, message: str):
        self.controller.report_error(message)

    def _apply_default_route_folder(self):
        route_dir = Path(self.controller.output_path) / "route"
        if not route_dir.is_dir():
            return
        self._selected_folder = str(route_dir)
        set_path_label(self._path_label, self._selected_folder)
        self._load_btn.setEnabled(True)

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select GraphNav Map Folder")
        if folder:
            self._selected_folder = folder
            set_path_label(self._path_label, folder)
            self._load_btn.setEnabled(True)

    def _on_load(self):
        if not self._selected_folder:
            return
        self.graph_widget.load_map(
            self._selected_folder,
            anchoring=self._anchoring_chk.isChecked(),
            show_waypoint_text=self._wp_text_chk.isChecked(),
            show_world_object_text=self._wo_text_chk.isChecked(),
        )

    def _on_clear(self):
        self.graph_widget.clear()
        if self.clear_uploaded_route:
            self.clear_uploaded_route()
            self.clear_uploaded_route = None

    def _on_start(self):
        if not self._selected_folder:
            self.controller.report_error("No folder selected")
            return

        self._start_btn.setEnabled(False)
        self.route_started.emit()

        self._route_worker = RouteWorker(
            self.controller,
            self._selected_folder,
            capture_interval_m=self._interval_spin.value(),
        )
        self._route_worker.finished.connect(self._on_route_finished)
        self._route_worker.error.connect(self._on_route_error)
        self._route_worker.start()

    def _on_route_finished(self, clear_fn):
        self.clear_uploaded_route = clear_fn
        self._start_btn.setEnabled(True)
        self.route_finished.emit()

    def _on_route_error(self, exc: Exception):
        self._start_btn.setEnabled(True)
        self.controller.report_error(str(exc), exc)
        self.route_finished.emit()

    def cleanup(self):
        self.graph_widget.cleanup_vtk()
