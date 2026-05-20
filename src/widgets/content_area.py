from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QStackedWidget,
    QFrame,
    QSizePolicy,
    QLineEdit,
)
from PyQt5.QtCore import Qt, QTimer

from utils.route.record_route import RecordingInterface
from utils.worker.record_worker import RecordWorker
from widgets.graph_nav_visualizer import GraphNavWidget

from utils.worker.route_worker import RouteWorker

import os

class ContentArea(QWidget):

    AUTO_PAGE = 0
    RECORD_PAGE = 1
    UPLOAD_PAGE = 2
    MANUAL_PAGE = 3

    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        layout = QVBoxLayout()

        self.stack = QStackedWidget()

        # Pages
        self.auto_page = self.create_page(
            "Auto Walk\n\n3D point cloud visualization"
        )

        self.record_page = self._create_record_page()

        # Upload page
        self.upload_page = self._create_upload_page()
        self.clear_uploaded_route = None

        self.manual_page = self.create_page(
            "Manual Walk\n\nManual robot control"
        )

        # Add pages
        self.stack.addWidget(self.auto_page)
        self.stack.addWidget(self.record_page)
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.manual_page)

        layout.addWidget(self.stack)

        self.setLayout(layout)

    def _create_upload_page(self):
        page = QWidget()
        main_layout = QHBoxLayout(page)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
 
        # LEFT: control panel
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        control_panel.setFixedWidth(220)
 
        ctrl = QVBoxLayout(control_panel)
        ctrl.setSpacing(10)
        ctrl.setContentsMargins(10, 10, 10, 10)
 
        ctrl.addWidget(QLabel("<b>GraphNav Map</b>"))
 
        # Path display
        self._path_label = QLabel("No folder selected")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: #aaa; font-size: 11px;")
 
        # Buttons
        browse_btn = QPushButton("Browse…")
        self._load_btn = QPushButton("Load Map")
        self._load_btn.setEnabled(False)
        clear_btn = QPushButton("Clear")
 
        # Options
        self._anchoring_chk = QCheckBox("Use anchoring")
        self._wp_text_chk = QCheckBox("Waypoint labels")
        self._wp_text_chk.setChecked(True)
        self._wo_text_chk = QCheckBox("World-object labels")
        self._wo_text_chk.setChecked(True)
 
        ctrl.addWidget(browse_btn)
        ctrl.addWidget(self._path_label)
 
        # Thin separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        ctrl.addWidget(sep)
 
        ctrl.addWidget(self._anchoring_chk)
        ctrl.addWidget(self._wp_text_chk)
        ctrl.addWidget(self._wo_text_chk)
        ctrl.addWidget(self._load_btn)
        ctrl.addWidget(clear_btn)
 
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        ctrl.addWidget(sep2)
 
        # Upload section
        ctrl.addWidget(QLabel("<b>Start Capturing</b>"))
        upload_btn = QPushButton("Start")
        ctrl.addWidget(upload_btn)
 
        ctrl.addStretch()
 
        # RIGHT: VTK visualizer
        self._graph_nav_widget = GraphNavWidget()
 
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self._graph_nav_widget, stretch=1)
 
        # Callbacks
        self._selected_folder: str | None = None
 
        def on_browse():
            folder = QFileDialog.getExistingDirectory(self, "Select GraphNav Map Folder")
            if folder:
                self._selected_folder = folder
                # Show shortened path so it fits in the panel
                display = folder if len(folder) <= 30 else "…" + folder[-28:]
                self._path_label.setText(display)
                self._path_label.setToolTip(folder)
                self._load_btn.setEnabled(True)
 
        def on_load():
            if not self._selected_folder:
                return
            self._graph_nav_widget.load_map(
                self._selected_folder,
                anchoring=self._anchoring_chk.isChecked(),
                show_waypoint_text=self._wp_text_chk.isChecked(),
                show_world_object_text=self._wo_text_chk.isChecked(),
            )
 
        def on_clear():
            self._graph_nav_widget.clear()
            if self.clear_uploaded_route:
                self.clear_uploaded_route()
                self.clear_uploaded_route = None
 
        def on_upload():
            if not self._selected_folder:
                self.controller.error_signal.emit("No folder selected")
                print("No folder selected")
                return
            
            upload_btn.setEnabled(False)
            
            self._route_worker = RouteWorker(self.controller, self._selected_folder)
            self._route_worker.finished.connect(on_route_finished)
            self._route_worker.error.connect(on_route_error)
            self._route_worker.start()

        def on_route_finished(clear_fn):
            self.clear_uploaded_route = clear_fn
            upload_btn.setEnabled(True)
            print("Route complete.")

        def on_route_error(e):
            upload_btn.setEnabled(True)
            self.controller.error_signal.emit(str(e))
            print(f"Route failed: {e}")
 
        browse_btn.clicked.connect(on_browse)
        self._load_btn.clicked.connect(on_load)
        clear_btn.clicked.connect(on_clear)
        upload_btn.clicked.connect(on_upload)
 
        return page


    def _create_record_page(self):

        page = QWidget()
        main_layout = QHBoxLayout(page)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # RIGHT: live visualizer
        self._record_graph_widget = GraphNavWidget(placeholder_text="")
        

        # Poll the robot graph every second while recording
        self._record_timer = QTimer()
        self._record_timer.setInterval(1000)

        def on_graph_refresh():
            try:
                graph = self.controller.graph_nav_client.download_graph()
                if graph is None:
                    return
                waypoints = {wp.id: wp for wp in graph.waypoints}
                # snapshots not available live, pass empty dict — axes + edges still render
                self._record_graph_widget.refresh(graph, waypoints, {})
            except Exception:
                pass  # robot may be mid-move, skip this tick

        self._record_timer.timeout.connect(on_graph_refresh)

        # LEFT: control panel
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        control_panel.setFixedWidth(220)

        ctrl = QVBoxLayout(control_panel)
        ctrl.setSpacing(10)
        ctrl.setContentsMargins(10, 10, 10, 10)

        ctrl.addWidget(QLabel("<b>Recording Settings</b>"))

        # Download path
        ctrl.addWidget(QLabel("Save Folder:"))
        self._record_path_label = QLabel("No folder selected")
        self._record_path_label.setWordWrap(True)
        self._record_path_label.setStyleSheet("font-size: 11px;")
        record_browse_btn = QPushButton("Browse...")
        ctrl.addWidget(record_browse_btn)
        ctrl.addWidget(self._record_path_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        ctrl.addWidget(sep)

        # Session name
        ctrl.addWidget(QLabel("Session Name:"))
        self._session_name_input = QLineEdit()
        self._session_name_input.setPlaceholderText("e.g. lab_run_01")
        ctrl.addWidget(self._session_name_input)

        # User name
        ctrl.addWidget(QLabel("Operator:"))
        self._user_name_input = QLineEdit()
        _user_name = self.controller.robot._current_user if self.controller.robot else "unknown"
        self._user_name_input.setText(_user_name)
        ctrl.addWidget(self._user_name_input)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        ctrl.addWidget(sep2)

        # Recording controls
        ctrl.addWidget(QLabel("<b>Recording</b>"))
        self._loop_chk = QCheckBox("Create Loop")
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

        self._record_selected_folder: None | str = None

        default_path = str(self.controller.output_path)
        self._record_selected_folder = default_path
        display = default_path if len(default_path) <= 30 else "..." + default_path[-28:]
        self._record_path_label.setText(display)
        self._record_path_label.setToolTip(default_path) 

        def on_browse():
            folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
            if folder:
                self._record_selected_folder = folder
                display = folder if len(folder) <= 30 else "..." + folder[-28:]
                self._record_path_label.setText(display)
                self._record_path_label.setToolTip(folder)

                if not self._session_name_input.text().strip():
                    self._session_name_input.setText(os.path.basename(folder))

        def on_start():
            if not self._record_selected_folder:
                self.controller.error_signal.emit("No save folder selected.")
                return

            session_name = self._session_name_input.text().strip() or os.path.basename(self._record_selected_folder)
            user_name = self._user_name_input.text().strip()

            self.controller.record_route_start(
                self._record_selected_folder,
                session_name,
                user_name,
                on_finished=lambda: (
                    self._start_btn.setEnabled(False),
                    self._waypoint_btn.setEnabled(True),
                    self._stop_btn.setEnabled(True),
                ),
                on_error=lambda e: self.controller.error_signal.emit(str(e))
            )
            self._record_timer.start()

        def on_waypoint():
            self.controller.record_route_waypoint(
                on_error=lambda e: self.controller.error_signal.emit(str(e))
            )

        def on_stop():
            self._stop_btn.setEnabled(False)
            self._waypoint_btn.setEnabled(False)
            self.controller.record_route_stop(
                create_loop=self._loop_chk.isChecked(),
                on_finished=lambda: self._start_btn.setEnabled(True),
                on_error=lambda e: self.controller.error_signal.emit(str(e))
            )
            self._record_timer.stop()

        record_browse_btn.clicked.connect(on_browse)
        self._start_btn.clicked.connect(on_start)
        self._waypoint_btn.clicked.connect(on_waypoint)
        self._stop_btn.clicked.connect(on_stop)



        main_layout.addWidget(control_panel)
        main_layout.addWidget(self._record_graph_widget, stretch=1)

        return page


    def create_page(self, text):
            page = QWidget()

            layout = QVBoxLayout()

            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)

            layout.addStretch()
            layout.addWidget(label)
            layout.addStretch()

            page.setLayout(layout)

            return page

    def show_auto_page(self):
        self.stack.setCurrentIndex(self.AUTO_PAGE)

    def show_record_page(self):
        self.stack.setCurrentIndex(self.RECORD_PAGE)

    def show_upload_page(self):
        self.stack.setCurrentIndex(self.UPLOAD_PAGE)

    def show_manual_page(self):
        self.stack.setCurrentIndex(self.MANUAL_PAGE)