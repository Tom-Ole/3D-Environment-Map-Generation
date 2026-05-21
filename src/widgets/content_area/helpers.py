from PyQt5.QtWidgets import QLabel, QFrame, QSizePolicy, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt


CONTENT_MARGIN = 12
CONTROL_PANEL_WIDTH = 260


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionHeader")
    return label


def muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("mutedLabel")
    label.setWordWrap(True)
    return label


def placeholder_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("placeholderLabel")
    label.setAlignment(Qt.AlignCenter)
    label.setWordWrap(True)
    return label


def separator() -> QFrame:
    sep = QFrame()
    sep.setObjectName("separator")
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Plain)
    sep.setFixedHeight(1)
    return sep


def control_panel() -> QFrame:
    panel = QFrame()
    panel.setObjectName("controlPanel")
    panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    panel.setFixedWidth(CONTROL_PANEL_WIDTH)
    return panel


def set_path_label(label: QLabel, path: str) -> None:
    display = path if len(path) <= 30 else "\u2026" + path[-28:]
    label.setText(display)
    label.setToolTip(path)


def map_page(control_panel: QFrame, main_widget: QWidget) -> QWidget:
    page = QWidget()
    layout = QHBoxLayout(page)
    layout.setContentsMargins(
        CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
    )
    layout.setSpacing(12)
    layout.addWidget(control_panel)
    layout.addWidget(main_widget, stretch=1)
    return page
