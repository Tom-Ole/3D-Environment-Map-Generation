from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QScrollArea,
    QGroupBox,
)

from utils.preprocess.mask import Preprocessor
from widgets.content_area.helpers import (
    section_label,
    muted_label,
    separator,
    CONTENT_MARGIN,
    set_path_label,
)


class PreprocessorPanel(QWidget):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._preprocessor = Preprocessor()
        self._input_folder: str | None = None
        self._output_folder: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN)
        layout.setSpacing(12)

        input_group = QGroupBox("Input Folder")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(8)
        
        input_row = QHBoxLayout()
        self._input_path_label = muted_label("No folder selected")
        input_browse_btn = QPushButton("Browse\u2026")
        input_row.addWidget(input_browse_btn)
        input_row.addWidget(self._input_path_label, stretch=1)
        input_layout.addLayout(input_row)
        layout.addWidget(input_group)

        output_group = QGroupBox("Output Folder")
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(8)
        
        output_row = QHBoxLayout()
        self._output_path_label = muted_label("No folder selected")
        output_browse_btn = QPushButton("Browse\u2026")
        output_row.addWidget(output_browse_btn)
        output_row.addWidget(self._output_path_label, stretch=1)
        output_layout.addLayout(output_row)
        layout.addWidget(output_group)

        class_group = QGroupBox("Class Selection")
        class_layout = QVBoxLayout(class_group)
        class_layout.setSpacing(8)
        
        class_scroll = QScrollArea()
        class_scroll.setWidgetResizable(True)
        class_scroll.setMaximumHeight(200)
        
        class_widget = QWidget()
        class_grid = QVBoxLayout(class_widget)
        class_grid.setSpacing(6)
        
        self._class_checkboxes = {}
        classes = self._preprocessor.get_classes()
        
        for class_id, class_name in classes.items():
            chk = QCheckBox(class_name)
            if class_id == 0:
                chk.setChecked(True)
            self._class_checkboxes[class_id] = chk
            class_grid.addWidget(chk)
        
        class_grid.addStretch()
        class_scroll.setWidget(class_widget)
        class_layout.addWidget(class_scroll)
        layout.addWidget(class_group)

        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)
        process_layout.setSpacing(8)
        
        self._start_btn = QPushButton("Start Preprocessing")
        self._start_btn.setEnabled(False)
        self._status_label = muted_label("Ready")
        process_layout.addWidget(self._start_btn)
        process_layout.addWidget(self._status_label)
        layout.addWidget(process_group)
        
        layout.addStretch()

        input_browse_btn.clicked.connect(self._on_input_browse)
        output_browse_btn.clicked.connect(self._on_output_browse)
        self._start_btn.clicked.connect(self._on_start)

    def _on_input_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Images)")
        if folder:
            self._input_folder = folder
            set_path_label(self._input_path_label, folder)
            self._update_start_button()

    def _on_output_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder (Masks)")
        if folder:
            self._output_folder = folder
            set_path_label(self._output_path_label, folder)
            self._update_start_button()

    def _update_start_button(self):
        self._start_btn.setEnabled(
            self._input_folder is not None and self._output_folder is not None
        )

    def _on_start(self):
        if not self._input_folder or not self._output_folder:
            self.controller.report_error("Please select both input and output folders.")
            return

        selected_classes = [
            class_id for class_id, chk in self._class_checkboxes.items() if chk.isChecked()
        ]

        if not selected_classes:
            self.controller.report_error("Please select at least one class.")
            return

        self._start_btn.setEnabled(False)
        self._status_label.setText("Processing...")

        try:
            self._preprocessor.create_masks_recursive(
                self._input_folder,
                self._output_folder,
                classes=selected_classes,
            )
            self._status_label.setText("Completed successfully.")
        except Exception as e:
            self.controller.report_error(f"Preprocessing failed: {e}", e)
            self._status_label.setText("Error occurred.")
        finally:
            self._start_btn.setEnabled(True)

    def cleanup(self):
        pass