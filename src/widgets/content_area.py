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
)
from PyQt5.QtCore import Qt
 
from widgets.graph_nav_visualizer import GraphNavWidget



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

        self.record_page = self.create_page(
            "Record Route\n\nRecording current route"
        )

        self.upload_page = self._create_upload_page()

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
        ctrl.addWidget(QLabel("<b>Upload to Robot</b>"))
        upload_btn = QPushButton("Upload")
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
 
        def on_upload():
            if not self._selected_folder:
                print("No folder selected")
                return
            print(f"Uploading: {self._selected_folder}")
            # self.controller.upload_route(self._selected_folder)
 
        browse_btn.clicked.connect(on_browse)
        self._load_btn.clicked.connect(on_load)
        clear_btn.clicked.connect(on_clear)
        upload_btn.clicked.connect(on_upload)
 
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