
from __future__ import annotations
 
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
 
 
 
@dataclass
class CaptureWaypoint:
    """One waypoint at which the robot should stop and capture images."""
 
    # GraphNav waypoint ID.
    waypoint_id: str
 
    label: str = ""
 
    # approximate distance (m) from the previous capture waypoint.
    distance_from_prev: float = 0.0
 
    # Arbitrary per-waypoint notes
    notes: str = ""
 
 
@dataclass
class RouteDefinition:
    """Complete route descriptor - serialised as route.json on disk."""
 
    # Directory that contains this file plus the GraphNav map data.
    route_dir: str
 
    # Ordered list of waypoints where images should be captured.
    capture_waypoints: List[CaptureWaypoint] = field(default_factory=list)
 
    # GraphNav waypoint ID used as the starting / localisation anchor.
    # The robot must be placed near this waypoint before running the route.
    seed_waypoint_id: str = ""
 
    description: str = ""
 
    # metres – minimum distance the robot must travel before a new automatic
    # capture waypoint is created during recording (0 = manual only).
    auto_capture_distance: float = 0.0
 

 
    def save(self) -> Path:
        """Write route.json to *route_dir*. Returns the path written."""
        path = Path(self.route_dir) / "route.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        return path
 
    @classmethod
    def load(cls, route_dir: str | Path) -> "RouteDefinition":
        """Load route.json from *route_dir*."""
        path = Path(route_dir) / "route.json"
        with open(path) as fh:
            data = json.load(fh)
        data["capture_waypoints"] = [
            CaptureWaypoint(**wp) for wp in data.get("capture_waypoints", [])
        ]
        return cls(**data)
 
 
    def add_capture_waypoint(
        self,
        waypoint_id: str,
        label: str = "",
        distance_from_prev: float = 0.0,
        notes: str = "",
    ) -> CaptureWaypoint:
        wp = CaptureWaypoint(
            waypoint_id=waypoint_id,
            label=label or f"wp_{len(self.capture_waypoints) + 1:03d}",
            distance_from_prev=distance_from_prev,
            notes=notes,
        )
        self.capture_waypoints.append(wp)
        return wp
 
    @property
    def waypoint_ids(self) -> List[str]:
        return [wp.waypoint_id for wp in self.capture_waypoints]
 
    def summary(self) -> str:
        lines = [
            f"Route: {self.route_dir}",
            f"  Description      : {self.description or '(none)'}",
            f"  Seed waypoint    : {self.seed_waypoint_id or '(not set)'}",
            f"  Capture waypoints: {len(self.capture_waypoints)}",
        ]
        for i, wp in enumerate(self.capture_waypoints):
            lines.append(
                f"    [{i:3d}] {wp.label:<20s}  id={wp.waypoint_id}  "
                f"dist_prev={wp.distance_from_prev:.2f} m"
                + (f"  notes={wp.notes!r}" if wp.notes else "")
            )
        return "\n".join(lines)
