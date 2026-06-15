"""Main GUI application for SPOT LiDAR capture and 3D reconstruction."""

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QStatusBar,
)
from PySide6.QtCore import Qt

from config import Config, load_config
from gui.tabs.capture_tab import CaptureTab
from gui.tabs.reconstruct_tab import ReconstructTab
from gui.tabs.ai_reconstruct_tab import AIReconstructTab

logger = logging.getLogger(__name__)


def get_modern_stylesheet() -> str:
    """
    Get a modern dark theme stylesheet for PySide6.

    Returns:
        CSS stylesheet as string
    """
    return """
    /* Main Window and Global */
    QMainWindow {
        background-color: #1e1e1e;
    }

    QWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }

    /* Tab Widget */
    QTabWidget::pane {
        border: 1px solid #3a3a3a;
        background-color: #1e1e1e;
    }

    QTabBar::tab {
        background-color: #2d2d2d;
        color: #a0a0a0;
        padding: 8px 20px;
        margin-right: 2px;
        border: 1px solid #3a3a3a;
        border-bottom: none;
    }

    QTabBar::tab:selected {
        background-color: #0d7377;
        color: #ffffff;
        border-bottom: 2px solid #0d7377;
    }

    QTabBar::tab:hover:!selected {
        background-color: #3a3a3a;
    }

    /* Push Buttons */
    QPushButton {
        background-color: #0d7377;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 11pt;
    }

    QPushButton:hover {
        background-color: #14919b;
    }

    QPushButton:pressed {
        background-color: #0a5a62;
    }

    QPushButton:disabled {
        background-color: #3a3a3a;
        color: #606060;
    }

    /* Group Boxes */
    QGroupBox {
        color: #ffffff;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        margin-top: 10px;
        padding-top: 10px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px 0 3px;
    }

    /* Line Edits */
    QLineEdit {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        padding: 6px;
        selection-background-color: #0d7377;
    }

    QLineEdit:focus {
        border: 2px solid #0d7377;
    }

    /* Text Edits */
    QTextEdit {
        background-color: #252525;
        color: #e0e0e0;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        padding: 6px;
    }

    QTextEdit:focus {
        border: 2px solid #0d7377;
    }

    /* Spin Boxes and Combo Boxes */
    QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        padding: 6px;
    }

    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border: 2px solid #0d7377;
    }

    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        background-color: #0d7377;
    }

    QComboBox::drop-down {
        background-color: #0d7377;
        border: none;
    }

    /* Labels */
    QLabel {
        color: #e0e0e0;
    }

    /* Status Bar */
    QStatusBar {
        background-color: #2d2d2d;
        color: #a0a0a0;
        border-top: 1px solid #3a3a3a;
    }

    /* Progress Bar */
    QProgressBar {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #3a3a3a;
        border-radius: 4px;
        padding: 2px;
    }

    QProgressBar::chunk {
        background-color: #0d7377;
        border-radius: 2px;
    }

    /* Scroll Bars */
    QScrollBar:vertical {
        background-color: #2d2d2d;
        width: 12px;
        border: none;
    }

    QScrollBar::handle:vertical {
        background-color: #3a3a3a;
        border-radius: 6px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #454545;
    }

    QScrollBar:horizontal {
        background-color: #2d2d2d;
        height: 12px;
        border: none;
    }

    QScrollBar::handle:horizontal {
        background-color: #3a3a3a;
        border-radius: 6px;
        min-width: 20px;
    }

    QScrollBar::handle:horizontal:hover {
        background-color: #454545;
    }

    /* Form Layout */
    QFormLayout {
        spacing: 10px;
    }

    /* Message Boxes */
    QMessageBox {
        background-color: #1e1e1e;
    }

    QMessageBox QLabel {
        color: #e0e0e0;
    }

    QMessageBox QPushButton {
        min-width: 60px;
    }
    """


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self, config: Config):
        """
        Initialize main window.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.capture_tab = None
        self.reconstruct_tab = None
        self.ai_reconstruct_tab = None

        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup main window UI."""
        self.setWindowTitle("SPOT 3D Capture & Reconstruction — LiDAR-SLAM + AI")
        self.setGeometry(100, 100, 1400, 900)

        # Create main widget
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Create tabs
        self.capture_tab = CaptureTab(self.config)
        self.reconstruct_tab = ReconstructTab(self.config)
        self.ai_reconstruct_tab = AIReconstructTab(self.config)

        self.tabs.addTab(self.capture_tab, "Capture")
        self.tabs.addTab(self.reconstruct_tab, "LiDAR-SLAM")
        self.tabs.addTab(self.ai_reconstruct_tab, "AI Reconstruction")

        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        # Apply stylesheet
        self.setStyleSheet(get_modern_stylesheet())

        logger.info("Main window initialized")

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Closing application")

        # Stop recording if active
        if self.capture_tab:
            self.capture_tab.stop_recording()

        event.accept()


def main():
    """Main application entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Starting SPOT LiDAR 3D Capture & Reconstruction application")

    try:
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded: hostname={config.robot_hostname}, output={config.output_dir}")

        # Create and run application
        app = QApplication(sys.argv)

        # Set application style
        app.setStyle("Fusion")

        # Create main window
        window = MainWindow(config)
        window.show()

        sys.exit(app.exec())

    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
