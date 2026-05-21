from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RobotStatusSnapshot:
    hostname: str = ""
    connected: bool = False
    battery_percent: Optional[float] = None
    motor_power: str = "Unknown"
    estop_active: bool = False
    lease_held: bool = False
    recording: bool = False
    session_path: str = ""
    lines: list[str] = field(default_factory=list)


def format_status_text(snapshot: RobotStatusSnapshot) -> str:
    return "\n".join(snapshot.lines)
