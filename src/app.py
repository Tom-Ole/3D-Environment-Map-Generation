import sys
import argparse

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QLineEdit,
    QPushButton, QVBoxLayout, QLabel, QFormLayout
)
from main_window import MainWindow

import bosdyn.client.util

from utils.controller.spot_controller import SpotController
from utils.controller.sim_spot_controller import SimSpotController


def login_prompt():
    dialog = QDialog()
    dialog.setWindowTitle("Authenticate SPOT SDK")

    layout = QVBoxLayout(dialog)
    form = QFormLayout()

    username_textbox = QLineEdit()
    password_textbox = QLineEdit()
    password_textbox.setEchoMode(QLineEdit.Password)  # hide password chars

    form.addRow(QLabel("Username:"), username_textbox)
    form.addRow(QLabel("Password:"), password_textbox)
    layout.addLayout(form)

    login_btn = QPushButton("Login")
    cancel_btn = QPushButton("Cancel")
    layout.addWidget(login_btn)
    layout.addWidget(cancel_btn)

    # Accept/reject the dialog on button click
    login_btn.clicked.connect(dialog.accept)
    cancel_btn.clicked.connect(dialog.reject)

    if dialog.exec_() == QDialog.Rejected:
        sys.exit(0)  # user cancelled — exit cleanly

    return username_textbox.text(), password_textbox.text()


def main():


    app = QApplication(sys.argv)

    options = get_args()


    try:
        with open("styles/style.qss", "r") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        with open("src/styles/style.qss", "r") as f:
            app.setStyleSheet(f.read())


    if(not options.isSim):
        sdk = bosdyn.client.create_standard_sdk("estop_gui")
        robot = sdk.create_robot(options.hostname)
        bosdyn.client.util.authenticate(robot, askpass=login_prompt)
        controller = SpotController(robot)
    else:
        #print(login_prompt())
        controller = SimSpotController()
    

    window = MainWindow(controller)
    window.show()

    sys.exit(app.exec_())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--hostname", help="Hostname of the robot", default="192.168.10.3")
    parser.add_argument("-S", "--isSim", help="Start a pseudo Simulation for debugging the GUI", action="store_true")

    #bosdyn.client.util.add_base_arguments(parser)

    return parser.parse_args()


if __name__ == "__main__":
    main()