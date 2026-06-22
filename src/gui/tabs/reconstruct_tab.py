"""Reconstruct tab UI."""

import logging
import subprocess
import sys
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
from gui.workers.colorize_worker import ColorizeWorker

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
        self.colorize_worker: Optional[ColorizeWorker] = None
        self.selected_session: Optional[Path] = None
        self._last_colored_ply: Optional[Path] = None

        self.setup_ui()

    def setup_ui(self) -> None:
        """Setup UI components."""
        main_layout = QHBoxLayout()

        # Control panel (left)
        control_panel = QGroupBox("LiDAR-SLAM Reconstruction Controls")
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

        self.colorize_btn = QPushButton("Colorize Mesh")
        self.colorize_btn.setToolTip(
            "Project camera images onto the reconstructed mesh vertices"
        )
        self.colorize_btn.clicked.connect(self.on_colorize_mesh)
        self.colorize_btn.setEnabled(False)
        button_layout.addWidget(self.colorize_btn)

        self.view_btn = QPushButton("View Results")
        self.view_btn.setToolTip("Open reconstruction results in Open3D viewer")
        self.view_btn.clicked.connect(self.on_view_results)
        self.view_btn.setEnabled(False)
        button_layout.addWidget(self.view_btn)

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
LiDAR-SLAM Reconstruction

This tab runs the LiDAR-based reconstruction pipeline using
KISS-ICP odometry, pose-graph optimisation, and Poisson meshing.

Steps performed:
1. Load LiDAR scans + SPOT poses from the selected session
2. KISS-ICP odometry (warm-started from SPOT VIO)
3. Loop closure detection (spatial KD-tree proximity)
4. Global pose-graph optimisation (Levenberg-Marquardt)
5. Point cloud fusion at optimised poses
6. Poisson surface mesh generation

Outputs (in session/reconstruction/):
  cloud_optimized.ply  — fused downsampled point cloud
  mesh.ply / mesh.obj  — Poisson surface mesh

For AI-based camera reconstruction, use the AI Reconstruction tab.

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
            # Enable post-processing buttons if prior outputs already exist
            recon_dir = self.selected_session / "reconstruction"
            has_mesh = (recon_dir / "mesh.ply").exists() or (recon_dir / "mesh.obj").exists()
            has_images = (self.selected_session / "images").exists()
            self.colorize_btn.setEnabled(has_mesh and has_images)
            self.view_btn.setEnabled(has_mesh)

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
        # Enable post-processing buttons if outputs exist
        if self.selected_session:
            recon_dir = self.selected_session / "reconstruction"
            has_mesh = (recon_dir / "mesh.ply").exists() or (recon_dir / "mesh.obj").exists()
            has_images = (self.selected_session / "images").exists()
            self.colorize_btn.setEnabled(has_mesh and has_images)
            self.view_btn.setEnabled(has_mesh)

    @Slot(str)
    def on_error(self, error_msg: str) -> None:
        """Handle reconstruction error."""
        logger.error(f"Reconstruction error: {error_msg}")
        self.log_display.append(f"\nError: {error_msg}")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Colorize
    # ------------------------------------------------------------------

    @Slot()
    def on_colorize_mesh(self) -> None:
        """Start mesh colorization in a background thread."""
        if not self.selected_session:
            return
        recon_dir = self.selected_session / "reconstruction"
        poses_file = recon_dir / "keyframe_poses.npy"
        if not poses_file.exists():
            self.log_display.append(
                "\nNo keyframe_poses.npy found — re-run reconstruction first."
            )
            return

        logger.info("Starting colorization…")
        self.log_display.append("\nStarting colorization…")
        self.colorize_btn.setEnabled(False)
        self.view_btn.setEnabled(False)

        self.colorize_worker = ColorizeWorker(session_path=self.selected_session)
        self.colorize_worker.finished.connect(self._on_colorize_finished)
        self.colorize_worker.error.connect(self._on_colorize_error)
        self.colorize_worker.log.connect(self.log_display.append)
        self.colorize_worker.start()

    @Slot(str)
    def _on_colorize_finished(self, ply_path: str) -> None:
        self._last_colored_ply = Path(ply_path)
        self.log_display.append(f"Coloured mesh → {ply_path}")
        self.colorize_btn.setEnabled(True)
        self.view_btn.setEnabled(True)

    @Slot(str)
    def _on_colorize_error(self, msg: str) -> None:
        self.log_display.append(f"\nColorization error: {msg}")
        self.colorize_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @Slot()
    def on_view_results(self) -> None:
        """Open reconstruction results in the Open3D standalone viewer."""
        if not self.selected_session:
            return
        recon_dir = self.selected_session / "reconstruction"

        # Prefer coloured mesh, then plain PLY, then OBJ
        candidates = [
            recon_dir / "mesh_colored.ply",
            recon_dir / "mesh.ply",
            recon_dir / "cloud_optimized.ply",
            recon_dir / "mesh.obj",
        ]
        target = next((p for p in candidates if p.exists()), None)
        if target is None:
            self.log_display.append("\nNo output file found to visualize.")
            return

        self.log_display.append(f"\nOpening viewer for: {target.name}")
        try:
            subprocess.Popen(
                [sys.executable, "-c",
                 f"import open3d as o3d; "
                 f"g = o3d.io.read_triangle_mesh(r'{target}') "
                 f"if r'{target}'.endswith(('.ply','.obj')) else "
                 f"o3d.io.read_point_cloud(r'{target}'); "
                 f"o3d.visualization.draw_geometries([g], "
                 f"window_name='{target.name}', width=1280, height=720)"],
                creationflags=getattr(__import__('subprocess'), 'CREATE_NO_WINDOW', 0),
            )
        except Exception as exc:
            self.log_display.append(f"Could not open viewer: {exc}")
