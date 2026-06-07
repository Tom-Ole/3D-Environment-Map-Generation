"""Reconstruct tab UI."""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
)
from PySide6.QtCore import Qt, Signal, Slot

from config import Config
from gui.workers.reconstruct_worker import ReconstructWorker

logger = logging.getLogger(__name__)


class ReconstructTab(QWidget):
    """Tab for offline reconstruction."""

    progress_updated = Signal(dict)

    def __init__(self, config: Config):
        """
        Initialize reconstruct tab.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.reconstruct_worker: Optional[ReconstructWorker] = None
        self.selected_session: Optional[Path] = None

        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup UI components."""
        main_layout = QHBoxLayout()

        # Control panel (left)
        control_panel = QGroupBox("Reconstruction Controls")
        control_layout = QVBoxLayout()

        # Session selection
        session_layout = QHBoxLayout()
        self.session_input = QLineEdit()
        self.session_input.setReadOnly(True)
        self.session_input.setPlaceholderText("Select a session folder...")
        session_layout.addWidget(self.session_input)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.on_browse_session)
        session_layout.addWidget(self.browse_btn)

        control_layout.addLayout(session_layout)

        # Parameters
        params_layout = QFormLayout()

        self.voxel_size_spinbox = QDoubleSpinBox()
        self.voxel_size_spinbox.setValue(0.05)
        self.voxel_size_spinbox.setMinimum(0.01)
        self.voxel_size_spinbox.setMaximum(1.0)
        self.voxel_size_spinbox.setSingleStep(0.01)
        params_layout.addRow("Voxel Size (m):", self.voxel_size_spinbox)

        self.loop_threshold_spinbox = QDoubleSpinBox()
        self.loop_threshold_spinbox.setValue(2.0)
        self.loop_threshold_spinbox.setMinimum(0.5)
        self.loop_threshold_spinbox.setMaximum(10.0)
        params_layout.addRow("Loop Threshold (m):", self.loop_threshold_spinbox)

        control_layout.addLayout(params_layout)

        # Buttons
        button_layout = QVBoxLayout()

        self.run_btn = QPushButton("Run Reconstruction")
        self.run_btn.clicked.connect(self.on_run_reconstruction)
        self.run_btn.setEnabled(False)
        button_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop_reconstruction)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        control_layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        # Log
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(250)
        control_layout.addWidget(self.log_display)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel, 1)

        # Info panel (right)
        info_panel = QGroupBox("Information")
        info_layout = QVBoxLayout()

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setText(
            """
3D Reconstruction

This tab allows you to:
1. Select a previously recorded session
2. Configure reconstruction parameters
3. Run the offline reconstruction pipeline
4. View progress and results

The pipeline performs:
- Odometry estimation (KISS-ICP)
- Loop closure detection
- Global pose graph optimization
- Point cloud fusion
- Mesh generation
- Results are saved in the session's reconstruction/ folder

Output directory: {}
        """.format(
                self.config.output_dir
            )
        )
        info_layout.addWidget(info_text)

        info_panel.setLayout(info_layout)
        main_layout.addWidget(info_panel, 1)

        self.setLayout(main_layout)

        logger.info("Reconstruct tab UI initialized")

    @Slot()
    def on_browse_session(self) -> None:
        """Browse for session folder."""
        session_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Session Folder",
            str(self.config.output_dir),
        )

        if session_dir:
            self.selected_session = Path(session_dir)
            self.session_input.setText(str(self.selected_session))
            self.run_btn.setEnabled(True)
            logger.info(f"Selected session: {self.selected_session}")

    @Slot()
    def on_run_reconstruction(self) -> None:
        """Start reconstruction."""
        if not self.selected_session:
            logger.warning("No session selected")
            return

        logger.info(f"Starting reconstruction for {self.selected_session}")

        # Create worker
        self.reconstruct_worker = ReconstructWorker(
            session_path=self.selected_session,
            voxel_size=self.voxel_size_spinbox.value(),
            loop_closure_threshold=self.loop_threshold_spinbox.value(),
        )

        self.reconstruct_worker.progress.connect(self.on_progress)
        self.reconstruct_worker.finished.connect(self.on_finished)
        self.reconstruct_worker.error.connect(self.on_error)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_display.clear()

        self.reconstruct_worker.start()

    @Slot()
    def on_stop_reconstruction(self) -> None:
        """Stop reconstruction."""
        if self.reconstruct_worker:
            logger.info("Stopping reconstruction")
            self.reconstruct_worker.stop()
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    @Slot(dict)
    def on_progress(self, progress_data: dict) -> None:
        """Handle progress update."""
        step = progress_data.get("step_name", "")
        pct = progress_data.get("progress_pct", 0)
        message = progress_data.get("message", "")

        self.progress_bar.setValue(int(pct))

        log_entry = f"[{step}] {pct:.0f}% - {message}\n"
        self.log_display.append(log_entry)

        # Auto-scroll to bottom
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    @Slot()
    def on_finished(self) -> None:
        """Handle reconstruction finished."""
        logger.info("Reconstruction finished")
        self.progress_bar.setValue(100)
        self.log_display.append("\nReconstruction complete!")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @Slot(str)
    def on_error(self, error_msg: str) -> None:
        """Handle error."""
        logger.error(f"Reconstruction error: {error_msg}")
        self.log_display.append(f"\nError: {error_msg}")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
