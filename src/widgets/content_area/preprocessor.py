from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QFileDialog,
)



class PreprocessorPanel(QWidget):

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

        