from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QFrame,
)
from PyQt5.QtCore import QTimer


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionHeader")
    return label


def _separator() -> QFrame:
    sep = QFrame()
    sep.setObjectName("separator")
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Plain)
    sep.setFixedHeight(1)
    return sep


class Sidebar(QWidget):

    _POLL_INTERVAL_MS = 2000

    def __init__(self, controller, hostname: str = ""):
        super().__init__()
        self.setObjectName("sidebar")

        self.controller = controller
        display_host = hostname or getattr(controller, "hostname", "") or "(unknown)"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(_section_label("Safety"))

        self.estop_btn = QPushButton("ESTOP")
        self.estop_btn.clicked.connect(self._estop)
        self.estop_btn.setObjectName("estopButton")

        self.release_btn = QPushButton("Release")
        self.release_btn.clicked.connect(self._release)
        self.release_btn.setDisabled(True)
        self.release_btn.setObjectName("releaseButton")

        layout.addWidget(self.estop_btn)
        layout.addWidget(self.release_btn)
        layout.addWidget(_separator())

        layout.addWidget(_section_label("Lease"))

        self.lease_btn = QPushButton("Release Lease")
        self.lease_btn.clicked.connect(self._release_lease)
        self.lease_btn.setObjectName("leaseButton")
        self.lease_btn.setEnabled(False)
        self.lease_btn.setToolTip(
            "Return the robot lease. Leases are taken automatically when running a route."
        )
        layout.addWidget(self.lease_btn)
        layout.addWidget(_separator())

        layout.addWidget(_section_label("Status"))

        self._host_label = QLabel(f"Robot: {display_host}")
        self._host_label.setObjectName("statusLabel")
        self._host_label.setWordWrap(True)
        layout.addWidget(self._host_label)

        self._connection_label = QLabel("Connection: …")
        self._connection_label.setObjectName("statusLabel")
        layout.addWidget(self._connection_label)

        self._battery_label = QLabel("Battery: …")
        self._battery_label.setObjectName("statusLabel")
        layout.addWidget(self._battery_label)

        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setPlaceholderText("Robot statistics\u2026")
        layout.addWidget(self.stats_box, stretch=1)

        self.setFixedWidth(240)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(self._POLL_INTERVAL_MS)
        self._refresh_status()

    def stop_polling(self):
        self._status_timer.stop()

    def _refresh_status(self):
        if not hasattr(self.controller, "get_status_snapshot"):
            return

        snapshot = self.controller.get_status_snapshot()

        host = snapshot.hostname or "(unknown)"
        self._host_label.setText(f"Robot: {host}")

        if snapshot.connected:
            self._connection_label.setText("Connection: Connected")
        else:
            self._connection_label.setText("Connection: Unavailable")

        if snapshot.battery_percent is not None:
            self._battery_label.setText(f"Battery: {snapshot.battery_percent:.0f}%")
        else:
            self._battery_label.setText("Battery: —")

        self.stats_box.setPlainText("\n".join(snapshot.lines))

        self._sync_estop_buttons(snapshot.estop_active)
        self._update_lease_button(snapshot.lease_held)

    def _sync_estop_buttons(self, estop_active: bool):
        self.estop_btn.setDisabled(estop_active)
        self.release_btn.setDisabled(not estop_active)

    def _estop(self):
        self.controller.estop()
        self._refresh_status()

    def _release(self):
        self.controller.release()
        self._refresh_status()

    def _update_lease_button(self, lease_held: bool | None = None):
        if lease_held is None:
            lease_held = self.controller.has_lease
        executing = getattr(self.controller, "is_executing_route", False)
        self.lease_btn.setEnabled(lease_held and not executing)

    def _release_lease(self):
        if self.controller.has_lease:
            self.controller.release_lease()
        self._refresh_status()
