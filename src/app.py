import sys
import argparse

from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

import bosdyn.client

from utils.controller.spot_controller import SpotController
from utils.controller.sim_spot_controller import SimSpotController


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
        controller = SpotController(robot)
    else:
        controller = SimSpotController()
    

    window = MainWindow(controller)
    window.show()

    sys.exit(app.exec_())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--hostname", help="Hostname of the robot", default="192.168.10.3")
    parser.add_argument("-S", "--isSim", help="Start a pseudo Simulation for debugging the GUI", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()