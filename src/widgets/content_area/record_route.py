import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QFileDialog,
)
from PyQt5.QtCore import QTimer

from widgets.graph_nav_visualizer import GraphNavWidget
from widgets.content_area.helpers import (
    section_label,
    muted_label,
    separator,
    control_panel,
    set_path_label,
    map_page,
)


class RecordRoutePanel(QWidget):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

        panel = control_panel()
        ctrl = QVBoxLayout(panel)
        ctrl.setContentsMargins(12, 12, 12, 12)
        ctrl.setSpacing(8)

        self.graph_widget = GraphNavWidget(placeholder_text="")
        self.graph_widget.load_failed.connect(self._on_error)

        self._record_timer = QTimer(self)
        self._record_timer.setInterval(1000)
        self._record_timer.timeout.connect(self._on_graph_refresh)

        ctrl.addWidget(section_label("Recording Settings"))
        ctrl.addWidget(QLabel("Save folder:"))
        self._path_label = muted_label("No folder selected")
        self._browse_btn = QPushButton("Browse\u2026")
        ctrl.addWidget(self._browse_btn)
        ctrl.addWidget(self._path_label)
        ctrl.addWidget(separator())

        ctrl.addWidget(QLabel("Session name:"))
        self._session_input = QLineEdit()
        self._session_input.setPlaceholderText("e.g. lab_run_01")
        ctrl.addWidget(self._session_input)

        ctrl.addWidget(QLabel("Operator:"))
        self._user_input = QLineEdit()
        user = (
            self.controller.robot._current_user
            if self.controller.robot
            else "unknown"
        )
        self._user_input.setText(user)
        ctrl.addWidget(self._user_input)
        ctrl.addWidget(separator())

        ctrl.addWidget(section_label("Recording"))
        self._loop_chk = QCheckBox("Create loop")
        ctrl.addWidget(self._loop_chk)

        self._start_btn = QPushButton("Start Recording")
        self._waypoint_btn = QPushButton("Create Waypoint")
        self._waypoint_btn.setEnabled(False)
        self._stop_btn = QPushButton("Stop & Save")
        self._stop_btn.setEnabled(False)

        ctrl.addWidget(self._start_btn)
        ctrl.addWidget(self._waypoint_btn)
        ctrl.addWidget(self._stop_btn)
        ctrl.addStretch()

        self._save_folder: str | None = None
        default_path = str(self.controller.output_path)
        self._save_folder = default_path
        set_path_label(self._path_label, default_path)
        self._session_input.setText(Path(self.controller.output_path).name)

        self._browse_btn.clicked.connect(self._on_browse)
        self._start_btn.clicked.connect(self._on_start)
        self._waypoint_btn.clicked.connect(self._on_waypoint)
        self._stop_btn.clicked.connect(self._on_stop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(map_page(panel, self.graph_widget))

    def _on_error(self, message: str):
        self.controller.report_error(message)

    def _on_graph_refresh(self):
        if self.controller.graph_nav_client is None:
            return
        try:
            graph = self.controller.graph_nav_client.download_graph()
            if graph is None:
                return
            waypoints = {wp.id: wp for wp in graph.waypoints}
            self.graph_widget.refresh(graph, waypoints, {})
        except Exception as e:
            self.controller.report_error(f"Graph refresh failed: {e}", e)

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self._save_folder = folder
            set_path_label(self._path_label, folder)
            self._session_input.setText(Path(folder).name)

    def _on_start(self):
        if not self._save_folder:
            self.controller.report_error("No save folder selected.")
            return

        session_name = (
            self._session_input.text().strip()
            or os.path.basename(self._save_folder)
        )
        user_name = self._user_input.text().strip()

        self.controller.record_route_start(
            self._save_folder,
            session_name,
            user_name,
            on_finished=self._on_start_finished,
            on_error=self._on_worker_error,
        )
        self._record_timer.start()

    def _on_start_finished(self):
        self._start_btn.setEnabled(False)
        self._waypoint_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)

    def _on_waypoint(self):
        self.controller.record_route_waypoint(on_error=self._on_worker_error)

    def _on_stop(self):
        self._stop_btn.setEnabled(False)
        self._waypoint_btn.setEnabled(False)
        self.controller.record_route_stop(
            create_loop=self._loop_chk.isChecked(),
            on_finished=self._on_stop_finished,
            on_error=self._on_worker_error,
        )
        self._record_timer.stop()

    def _on_stop_finished(self):
        self._start_btn.setEnabled(True)

    def _on_worker_error(self, exc: Exception):
        self.controller.report_error(str(exc), exc)

    def cleanup(self):
        self._record_timer.stop()
        self.graph_widget.cleanup_vtk()
