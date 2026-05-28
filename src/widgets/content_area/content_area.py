from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from widgets.content_area.coming_soon import build_coming_soon_panel
from widgets.content_area.record_route import RecordRoutePanel
from widgets.content_area.run_route import RunRoutePanel
from widgets.content_area.manual_walk import ManualWalkPanel
from widgets.content_area.preprocessor import PreprocessorPanel


class ContentArea(QWidget):

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("mainTabs")
        self._tabs.addTab(self._build_data_capture_tab(), "\U0001f4f7 Data Capture")
        
        self._preprocessor_panel = PreprocessorPanel(self.controller)
        self._tabs.addTab(self._preprocessor_panel, "\U0001f5bc Preprocessing")
        self._tabs.addTab(
            build_coming_soon_panel(
                "3D Reconstruction",
                "COLMAP reconstruction and 3D model viewing.",
            ),
            "\U0001f9ca 3D Reconstruction",
        )
        layout.addWidget(self._tabs)

        self.record_panel = self._record_panel
        self.run_panel = self._run_panel
        self.manual_walk_panel = self._manual_walk_panel
        self.preprocessor_panel = self._preprocessor_panel

    def _build_data_capture_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        sub_tabs = QTabWidget()
        sub_tabs.setObjectName("captureTabs")
        sub_tabs.addTab(
            build_coming_soon_panel(
                "Auto Walk",
                "Autonomous navigation with periodic image capture.",
            ),
            "Auto Walk",
        )

        self._record_panel = RecordRoutePanel(self.controller)
        sub_tabs.addTab(self._record_panel, "Record Route")

        self._run_panel = RunRoutePanel(self.controller)
        sub_tabs.addTab(self._run_panel, "Run Route")

        self._manual_walk_panel = ManualWalkPanel(self.controller)
        sub_tabs.addTab(self._manual_walk_panel, "Manual Walk")

        layout.addWidget(sub_tabs)
        return page

    def cleanup(self):
        self.record_panel.cleanup()
        self.run_panel.cleanup()
        self.manual_walk_panel.cleanup()
        self.preprocessor_panel.cleanup()
