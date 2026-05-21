from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from PyQt5.QtGui import QFont

from widgets.content_area.helpers import muted_label, CONTENT_MARGIN


def build_coming_soon_panel(title: str, description: str) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN)

    group = QGroupBox(title)
    group.setEnabled(False)
    inner = QVBoxLayout(group)
    inner.setSpacing(8)

    desc = muted_label(description)
    desc_font = QFont()
    desc_font.setItalic(True)
    desc.setFont(desc_font)
    inner.addWidget(desc)

    note = muted_label("This feature is not yet implemented.")
    note_font = QFont()
    note_font.setItalic(True)
    note.setFont(note_font)
    inner.addWidget(note)

    layout.addWidget(group)
    layout.addStretch()
    return page
