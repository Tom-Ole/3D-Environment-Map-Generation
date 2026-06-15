"""AI Reconstruction tab — camera-based 3D reconstruction via AI models."""

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import Config
from gui.workers.ai_reconstruct_worker import AIReconstructWorker

logger = logging.getLogger(__name__)

# Camera sources captured by SPOT's five fisheye cameras
_ALL_SOURCES = [
    ("frontleft_fisheye_image",  "Front-left fisheye",  True),
    ("frontright_fisheye_image", "Front-right fisheye", True),
    ("left_fisheye_image",       "Left fisheye",        False),
    ("right_fisheye_image",      "Right fisheye",       False),
    ("back_fisheye_image",       "Back fisheye",        False),
]

_MODEL_OPTIONS = [
    ("auto",      "Auto (best available)"),
    ("mast3r",    "MASt3R  — metric scale, ECCV 2024  [GPU recommended]"),
    ("dust3r",    "DUSt3R  — dense unconstrained,  CVPR 2024  [GPU recommended]"),
    ("vggt",      "VGGT    — single-pass transformer, CVPR 2025  [GPU recommended]"),
    ("geometric", "Geometric SfM  — ORB triangulation, CPU, no download"),
]

_DEVICE_OPTIONS = [
    ("auto", "Auto (detect CUDA / MPS / CPU)"),
    ("cuda", "CUDA  (NVIDIA GPU)"),
    ("mps",  "MPS   (Apple Silicon)"),
    ("cpu",  "CPU   (slow for neural models)"),
]

_IMAGE_SIZES = [224, 384, 512, 768, 1024]


class AIReconstructTab(QWidget):
    """Tab that runs the AI-based camera reconstruction pipeline."""

    progress_updated = Signal(dict)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._worker: Optional[AIReconstructWorker] = None
        self._selected_session: Optional[Path] = None
        self._source_checkboxes: List[QCheckBox] = []

        self._build_ui()
        self._check_model_availability()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ── Left panel: controls ──────────────────────────────────────────────
        controls = QGroupBox("AI Reconstruction Controls")
        ctrl_layout = QVBoxLayout()

        # Session selection
        sess_row = QHBoxLayout()
        self._session_input = QLineEdit()
        self._session_input.setReadOnly(True)
        self._session_input.setPlaceholderText("Select a recorded session folder…")
        sess_row.addWidget(self._session_input)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        sess_row.addWidget(browse_btn)
        ctrl_layout.addLayout(sess_row)

        # Parameters form
        form = QFormLayout()

        # Model
        self._model_combo = QComboBox()
        for key, label in _MODEL_OPTIONS:
            self._model_combo.addItem(label, userData=key)
        form.addRow("Model:", self._model_combo)

        # Device
        self._device_combo = QComboBox()
        for key, label in _DEVICE_OPTIONS:
            self._device_combo.addItem(label, userData=key)
        form.addRow("Device:", self._device_combo)

        # Image size
        self._size_combo = QComboBox()
        for s in _IMAGE_SIZES:
            self._size_combo.addItem(f"{s} px", userData=s)
        self._size_combo.setCurrentIndex(_IMAGE_SIZES.index(512))
        form.addRow("Image size:", self._size_combo)

        # Max images
        self._max_images_spin = QSpinBox()
        self._max_images_spin.setRange(4, 500)
        self._max_images_spin.setValue(100)
        self._max_images_spin.setSuffix(" frames")
        form.addRow("Max keyframes:", self._max_images_spin)

        # Keyframe strategy
        self._strategy_combo = QComboBox()
        self._strategy_combo.addItem("Interval (every N frames)", userData="interval")
        self._strategy_combo.addItem("Motion (translation / rotation threshold)", userData="motion")
        self._strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        form.addRow("Keyframe strategy:", self._strategy_combo)

        # Interval spinbox (visible when strategy = interval)
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 50)
        self._interval_spin.setValue(5)
        self._interval_label = QLabel("Keyframe interval:")
        form.addRow(self._interval_label, self._interval_spin)

        # Motion thresholds (visible when strategy = motion)
        self._min_trans_spin = QDoubleSpinBox()
        self._min_trans_spin.setRange(0.05, 5.0)
        self._min_trans_spin.setSingleStep(0.05)
        self._min_trans_spin.setValue(0.30)
        self._min_trans_spin.setSuffix(" m")
        self._min_trans_label = QLabel("Min translation:")
        form.addRow(self._min_trans_label, self._min_trans_spin)

        self._min_rot_spin = QDoubleSpinBox()
        self._min_rot_spin.setRange(1.0, 90.0)
        self._min_rot_spin.setSingleStep(1.0)
        self._min_rot_spin.setValue(10.0)
        self._min_rot_spin.setSuffix(" °")
        self._min_rot_label = QLabel("Min rotation:")
        form.addRow(self._min_rot_label, self._min_rot_spin)

        # Voxel downsample
        self._voxel_spin = QDoubleSpinBox()
        self._voxel_spin.setRange(0.0, 1.0)
        self._voxel_spin.setSingleStep(0.01)
        self._voxel_spin.setValue(0.05)
        self._voxel_spin.setSuffix(" m  (0 = off)")
        form.addRow("Voxel size:", self._voxel_spin)

        # Confidence threshold (DUSt3R / MASt3R)
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.0, 10.0)
        self._conf_spin.setSingleStep(0.1)
        self._conf_spin.setValue(1.5)
        form.addRow("Confidence threshold:", self._conf_spin)

        ctrl_layout.addLayout(form)

        # Camera source checkboxes
        src_box = QGroupBox("Camera Sources")
        src_layout = QVBoxLayout()
        for source_id, label, default in _ALL_SOURCES:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.setProperty("source_id", source_id)
            src_layout.addWidget(cb)
            self._source_checkboxes.append(cb)
        src_box.setLayout(src_layout)
        ctrl_layout.addWidget(src_box)

        # Run / Stop buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run AI Reconstruction")
        self._run_btn.clicked.connect(self._on_run)
        self._run_btn.setEnabled(False)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._stop_btn)
        ctrl_layout.addLayout(btn_row)

        # Stage label + overall progress bar
        self._stage_label = QLabel("Stage: —")
        self._stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl_layout.addWidget(self._stage_label)

        self._overall_bar = QProgressBar()
        self._overall_bar.setRange(0, 100)
        self._overall_bar.setValue(0)
        self._overall_bar.setFormat("Overall: %p%")
        ctrl_layout.addWidget(self._overall_bar)

        # Stage progress bar
        self._stage_bar = QProgressBar()
        self._stage_bar.setRange(0, 100)
        self._stage_bar.setValue(0)
        self._stage_bar.setFormat("Stage: %p%")
        ctrl_layout.addWidget(self._stage_bar)

        # Log display
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(200)
        ctrl_layout.addWidget(self._log)

        controls.setLayout(ctrl_layout)
        root.addWidget(controls, 3)

        # ── Right panel: info ─────────────────────────────────────────────────
        info_panel = QGroupBox("Model Availability & Information")
        info_layout = QVBoxLayout()

        self._avail_label = QLabel("Checking model availability…")
        self._avail_label.setWordWrap(True)
        self._avail_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        info_layout.addWidget(self._avail_label)

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMinimumHeight(300)
        info_text.setText(self._info_text())
        info_layout.addWidget(info_text)

        info_panel.setLayout(info_layout)
        root.addWidget(info_panel, 2)

        # Initial state of motion-threshold rows
        self._on_strategy_changed(0)

    # ── Slot handlers ─────────────────────────────────────────────────────────

    @Slot()
    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Session Folder", str(self.config.output_dir)
        )
        if folder:
            self._selected_session = Path(folder)
            self._session_input.setText(folder)
            self._run_btn.setEnabled(True)
            logger.info(f"AI session selected: {folder}")

    @Slot(int)
    def _on_strategy_changed(self, index: int) -> None:
        is_interval = (self._strategy_combo.currentData() == "interval")
        self._interval_label.setVisible(is_interval)
        self._interval_spin.setVisible(is_interval)
        self._min_trans_label.setVisible(not is_interval)
        self._min_trans_spin.setVisible(not is_interval)
        self._min_rot_label.setVisible(not is_interval)
        self._min_rot_spin.setVisible(not is_interval)

    @Slot()
    def _on_run(self) -> None:
        if not self._selected_session:
            return

        sources = [
            cb.property("source_id")
            for cb in self._source_checkboxes
            if cb.isChecked()
        ]
        if not sources:
            self._append_log("Select at least one camera source.")
            return

        self._worker = AIReconstructWorker(
            session_path=self._selected_session,
            model_type=self._model_combo.currentData(),
            device=self._device_combo.currentData(),
            image_size=self._size_combo.currentData(),
            keyframe_strategy=self._strategy_combo.currentData(),
            keyframe_interval=self._interval_spin.value(),
            keyframe_min_translation=self._min_trans_spin.value(),
            keyframe_min_rotation_deg=self._min_rot_spin.value(),
            max_images=self._max_images_spin.value(),
            camera_sources=sources,
            voxel_size=self._voxel_spin.value(),
            confidence_threshold=self._conf_spin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._overall_bar.setValue(0)
        self._stage_bar.setValue(0)
        self._stage_label.setText("Stage: starting…")
        self._log.clear()
        self._append_log(
            f"Starting AI reconstruction\n"
            f"  Session : {self._selected_session}\n"
            f"  Model   : {self._model_combo.currentText()}\n"
            f"  Device  : {self._device_combo.currentData()}\n"
            f"  Cameras : {', '.join(sources)}\n"
        )
        self._worker.start()

    @Slot()
    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self._append_log("\nStop requested — waiting for current stage to finish…")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    @Slot(dict)
    def _on_progress(self, data: dict) -> None:
        stage = data.get("stage", "")
        stage_idx = data.get("stage_index", 0)
        total = data.get("total_stages", 6)
        stage_pct = float(data.get("stage_pct", 0))
        overall_pct = float(data.get("overall_pct", 0))
        msg = data.get("message", "")

        self._stage_label.setText(
            f"Stage {stage_idx}/{total}: {stage.replace('_', ' ').title()}"
        )
        self._stage_bar.setValue(int(stage_pct))
        self._overall_bar.setValue(int(overall_pct))

        if stage_pct in (0.0, 100.0) or msg:
            self._append_log(
                f"[{stage_idx}/{total}] {stage}  {stage_pct:.0f}%  —  {msg}"
            )

    @Slot(dict)
    def _on_finished(self, stats: dict) -> None:
        pts = stats.get("point_count", 0)
        model = stats.get("model_used", "?")
        device = stats.get("device_used", "?")
        dur = stats.get("duration_seconds", 0.0)
        pcd_path = stats.get("point_cloud_path", "")

        self._overall_bar.setValue(100)
        self._stage_bar.setValue(100)
        self._stage_label.setText("Complete")
        self._append_log(
            f"\nAI Reconstruction complete!\n"
            f"  Points  : {pts:,}\n"
            f"  Model   : {model}\n"
            f"  Device  : {device}\n"
            f"  Time    : {dur:.1f} s\n"
            + (f"  Output  : {pcd_path}\n" if pcd_path else "")
        )
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        logger.error(f"AI reconstruction error: {msg}")
        self._append_log(f"\nError: {msg}")
        self._stage_label.setText("Error")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _append_log(self, text: str) -> None:
        self._log.append(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _check_model_availability(self) -> None:
        """Populate the availability label by probing each model's is_available()."""
        try:
            from ai_reconstruction.models import get_available_models
            rows = []
            for name, available, desc in get_available_models():
                icon = "OK" if available else "--"
                rows.append(f"  [{icon}]  {name:10s}  {desc}")
            self._avail_label.setText(
                "Model availability (detected at startup):\n\n" + "\n".join(rows)
            )
        except Exception as e:
            self._avail_label.setText(f"Could not check availability: {e}")

    @staticmethod
    def _info_text() -> str:
        return """\
AI Reconstruction Pipeline
==========================

This tab reconstructs 3D geometry from the SPOT robot's camera images
using state-of-the-art AI methods.  It is fully independent from the
LiDAR-SLAM pipeline on the other tab.

HOW IT WORKS
------------
1. Images are loaded from the selected session's images/ folder.
2. A keyframe subset is selected (interval- or motion-based).
3. An AI model predicts 3D point maps from image pairs.
4. The raw cloud is filtered, downsampled, and exported.

MODELS
------
MASt3R   Best quality; metric scale output; sparse global alignment.
          ~800 MB download.  Requires GPU (8+ GB VRAM).

DUSt3R   Dense reconstruction per image pair; global alignment via
          iterative bundle adjustment.  ~600 MB.  GPU recommended.

VGGT     Single forward pass — very fast; less accurate on large scenes.
          ~4 GB for 1B model.  GPU required.

Geometric  ORB triangulation using known SPOT VIO poses.  CPU only,
           no download.  Sparse but metric and always available.

OUTPUTS
-------
Results are saved in:
  <session>/ai_reconstruction/
    point_cloud.ply    — coloured point cloud
    camera_poses.npy   — Mx4x4 camera-to-world matrices
    metadata.json      — run statistics and configuration

TIPS
----
- Use frontleft + frontright cameras for best forward-scene coverage.
- Set keyframe interval = 3-5 for dense coverage, 10-15 for speed.
- For large sessions (> 100 frames), use motion-based selection.
- Geometric model works offline with no internet connection.
"""
