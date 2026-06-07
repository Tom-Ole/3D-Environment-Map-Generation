"""Capture tab UI for live SPOT sensor data recording."""

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
    QTextEdit,
    QDoubleSpinBox,
    QFormLayout,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot

from config import Config
from gui.workers.capture_worker import CaptureWorker

logger = logging.getLogger(__name__)


class CaptureTab(QWidget):
    """Tab for live LiDAR + camera + pose capture."""

    status_updated = Signal(dict)

    def __init__(self, config: Config):
        """
        Initialize capture tab.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.capture_worker: Optional[CaptureWorker] = None
        self.recording = False
        self.connected = False

        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup UI components."""
        main_layout = QHBoxLayout()

        # Control panel (left)
        control_panel = QGroupBox("Capture Controls")
        control_layout = QVBoxLayout()

        # Connection settings
        settings_layout = QFormLayout()

        self.hostname_input = QLineEdit(self.config.robot_hostname)
        settings_layout.addRow("Robot Hostname:", self.hostname_input)

        self.lidar_rate = QDoubleSpinBox()
        self.lidar_rate.setValue(10.0)
        self.lidar_rate.setMinimum(0.1)
        self.lidar_rate.setMaximum(30.0)
        self.lidar_rate.setSingleStep(1.0)
        settings_layout.addRow("LiDAR Rate (Hz):", self.lidar_rate)

        self.camera_rate = QDoubleSpinBox()
        self.camera_rate.setValue(5.0)
        self.camera_rate.setMinimum(0.1)
        self.camera_rate.setMaximum(30.0)
        self.camera_rate.setSingleStep(1.0)
        settings_layout.addRow("Camera Rate (Hz):", self.camera_rate)

        control_layout.addLayout(settings_layout)

        # Connection status
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        control_layout.addWidget(self.status_label)

        # Buttons
        button_layout = QVBoxLayout()

        self.connect_btn = QPushButton("Connect to Robot")
        self.connect_btn.clicked.connect(self.on_connect)
        button_layout.addWidget(self.connect_btn)

        self.start_btn = QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.on_start_recording)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Recording")
        self.stop_btn.clicked.connect(self.on_stop_recording)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        control_layout.addLayout(button_layout)

        # Statistics display
        stats_label = QLabel("Capture Statistics:")
        stats_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(stats_label)

        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMinimumHeight(200)
        self.stats_display.setMaximumHeight(250)
        control_layout.addWidget(self.stats_display)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel, 1)

        # Info panel (right)
        info_panel = QGroupBox("Information")
        info_layout = QVBoxLayout()

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setText(
            f"""
SPOT LiDAR Data Capture

This tab allows you to:
1. Connect to a SPOT robot
2. Configure sensor sampling rates
3. Start/stop recording of:
   • LiDAR point clouds (Velodyne-16)
   • Camera images (5 fisheye cameras)
   • Robot poses and IMU data

Recording Details:
• Data is saved automatically during capture
• Sessions are timestamped: recordings/YYYYMMDD_HHMMSS/
• No motion lease required (read-only operation)

After recording, use the Reconstruct tab to run
the offline 3D reconstruction pipeline.

Output Directory: {self.config.output_dir}
        """
        )
        info_layout.addWidget(info_text)

        info_panel.setLayout(info_layout)
        main_layout.addWidget(info_panel, 1)

        self.setLayout(main_layout)

        # Setup status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.setInterval(1000)  # Update every 1 second

        logger.info("Capture tab UI initialized")

    @Slot()
    def on_connect(self) -> None:
        """Handle connect button click."""
        if self.connected:
            logger.info("Already connected")
            return

        hostname = self.hostname_input.text()
        if not hostname:
            self.status_label.setText("Error: Empty hostname")
            self.status_label.setStyleSheet("color: red;")
            return

        logger.info(f"Connecting to {hostname}...")
        self.status_label.setText("Connecting...")
        self.status_label.setStyleSheet("color: orange;")
        self.connect_btn.setEnabled(False)
        self.hostname_input.setEnabled(False)

        # Create and start capture worker
        self.capture_worker = CaptureWorker(
            hostname=hostname,
            username=self.config.robot_username,
            password=self.config.robot_password,
            output_dir=self.config.output_dir,
            lidar_rate_hz=self.lidar_rate.value(),
            camera_rate_hz=self.camera_rate.value(),
        )

        self.capture_worker.connected.connect(self.on_connected)
        self.capture_worker.disconnected.connect(self.on_disconnected)
        self.capture_worker.error.connect(self.on_error)

        self.capture_worker.start()

    @Slot()
    def on_connected(self) -> None:
        """Handle successful connection."""
        logger.info("Connected to robot")
        self.connected = True
        self.status_label.setText("Connected")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.start_btn.setEnabled(True)
        self.connect_btn.setEnabled(False)
        self.lidar_rate.setEnabled(False)
        self.camera_rate.setEnabled(False)
        self.hostname_input.setEnabled(False)

        # Start status updates
        self.status_timer.start()

    @Slot()
    def on_disconnected(self) -> None:
        """Handle disconnection."""
        logger.info("Disconnected from robot")
        self.connected = False
        self.recording = False
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.connect_btn.setEnabled(True)
        self.lidar_rate.setEnabled(True)
        self.camera_rate.setEnabled(True)
        self.hostname_input.setEnabled(True)
        self.status_timer.stop()
        self.stats_display.clear()

    @Slot()
    def on_start_recording(self) -> None:
        """Handle start recording button."""
        if not self.connected or not self.capture_worker:
            logger.warning("Not connected")
            return

        logger.info("Starting recording")
        self.capture_worker.start_recording()
        self.recording = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.lidar_rate.setEnabled(False)
        self.camera_rate.setEnabled(False)

    @Slot()
    def on_stop_recording(self) -> None:
        """Handle stop recording button."""
        if not self.recording or not self.capture_worker:
            return

        logger.info("Stopping recording")
        self.capture_worker.stop_recording()
        self.recording = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.lidar_rate.setEnabled(True)
        self.camera_rate.setEnabled(True)

    @Slot(str)
    def on_error(self, error_msg: str) -> None:
        """Handle error from capture worker."""
        logger.error(f"Capture error: {error_msg}")
        self.status_label.setText(f"Error: {error_msg[:50]}")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")

    @Slot()
    def update_status(self) -> None:
        """Update statistics display from worker."""
        if self.capture_worker:
            stats = self.capture_worker.get_stats()
            stats_text = (
                f"LiDAR frames:  {stats.get('lidar_count', 0)}\n"
                f"Camera images: {stats.get('camera_count', 0)}\n"
                f"Pose updates:  {stats.get('pose_count', 0)}\n"
                f"Duration:      {stats.get('duration_sec', 0):.1f}s\n"
                f"Session:       {stats.get('session_id', 'N/A')}\n"
                f"Recording:     {'Yes' if stats.get('recording') else 'No'}"
            )
            self.stats_display.setText(stats_text)

    def stop_recording(self) -> None:
        """Stop recording and cleanup (called on app close)."""
        if self.recording:
            self.on_stop_recording()

        if self.capture_worker:
            if self.capture_worker.isRunning():
                self.capture_worker.running = False
                self.capture_worker.quit()
                self.capture_worker.wait(timeout=5000)
