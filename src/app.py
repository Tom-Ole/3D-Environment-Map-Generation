#import cv2 # Can Fix: "ImportError: libGL.so.1: cannot open shared object file: No such file or directory" on some Linux distros
import torch # Can Fix: "OSError: [WinError 1114] Eine DLL-Initialisierungsroutine ist fehlgeschlagen. Error loading '...\Lib\site-packages\torch\lib\c10.dll' or one of its dependencies."

import logging
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

import os
from dotenv import load_dotenv


# TODO: more research sublease
# TODO: Getting ESTOP right / returning / closing
# TODO: time it takes for preprocessing would be intresting
# TODO: Fix folder structure


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
        sys.exit(0)  # user cancelled - exit cleanly

    return username_textbox.text(), password_textbox.text()

def login_env():
    username = os.environ.get("BOSDYN_USERNAME")
    password = os.environ.get("BOSDYN_PASSWORD")
    return username, password


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    app = QApplication(sys.argv)

    options = get_args()


    try:
        with open("styles/style.qss", "r") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        try:
            with open("src/styles/style.qss", "r") as f:
                app.setStyleSheet(f.read())
        except FileNotFoundError:
            pass


    if(not options.isSim):
        sdk = bosdyn.client.create_standard_sdk("auto_3D")
        robot = sdk.create_robot(options.hostname)
        if not options.autologin:
            bosdyn.client.util.authenticate(robot, askpass=login_prompt)
        else:
            load_dotenv()
            bosdyn.client.util.authenticate(robot, askpass=login_env)

        controller = SpotController(robot, hostname=options.hostname)
    else:
        controller = SimSpotController()
    

    hostname = "" if options.isSim else options.hostname
    window = MainWindow(controller, hostname=hostname)
    window.show()

    sys.exit(app.exec_())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--hostname", help="Hostname of the robot", default="192.168.10.3")
    parser.add_argument("-S", "--isSim", help="Start a pseudo Simulation for debugging the GUI", action="store_true")
    parser.add_argument("-A", "--autologin", help="Login via .env variables", action="store_true")

    #bosdyn.client.util.add_base_arguments(parser)

    return parser.parse_args()


if __name__ == "__main__":
    main()