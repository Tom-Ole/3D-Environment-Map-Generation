"""
GraphNav map visualizer widget.

Embeds a VTK renderer into a PyQt5 widget using QVTKRenderWindowInteractor.
Extracted and adapted from the Boston Dynamics Spot SDK example.
"""

import os

import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

from bosdyn.api import geometry_pb2
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.math_helpers import SE3Pose


# ---------------------------------------------------------------------------
# Low-level VTK helpers (stateless, no Qt dependency)
# ---------------------------------------------------------------------------

def _numpy_to_poly_data(pts):
    """Convert a (N x 3) numpy array to vtkPolyData."""
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk.vtkPoints())
    pd.GetPoints().SetData(numpy_support.numpy_to_vtk(pts.copy()))
    f = vtk.vtkVertexGlyphFilter()
    f.SetInputData(pd)
    f.Update()
    return f.GetOutput()


def _mat_to_vtk(mat):
    """4x4 numpy matrix → vtkTransform."""
    t = vtk.vtkTransform()
    t.SetMatrix(mat.flatten())
    return t


def _vtk_to_mat(transform):
    """vtkTransform → 4x4 numpy matrix."""
    tf_matrix = transform.GetMatrix()
    out = np.eye(4)
    for r in range(4):
        for c in range(4):
            out[r, c] = tf_matrix.GetElement(r, c)
    return out


def _api_to_vtk_se3_pose(se3_pose):
    return _mat_to_vtk(se3_pose.to_matrix())


# ---------------------------------------------------------------------------
# Scene-object factories
# ---------------------------------------------------------------------------

def _make_line(pt_a, pt_b, renderer):
    src = vtk.vtkLineSource()
    src.SetPoint1(*pt_a[:3])
    src.SetPoint2(*pt_b[:3])
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(src.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2)
    actor.GetProperty().SetColor(0.7, 0.7, 0.7)
    renderer.AddActor(actor)
    return actor


def _make_text(name, pt, renderer):
    actor = vtk.vtkTextActor()
    actor.SetInput(name)
    prop = actor.GetTextProperty()
    prop.SetBackgroundColor(0.0, 0.0, 0.0)
    prop.SetBackgroundOpacity(0.5)
    prop.SetFontSize(14)
    coord = actor.GetPositionCoordinate()
    coord.SetCoordinateSystemToWorld()
    coord.SetValue(float(pt[0]), float(pt[1]), float(pt[2]))
    renderer.AddActor(actor)
    return actor


def _create_point_cloud_actor(waypoints, snapshots, waypoint_id):
    wp = waypoints[waypoint_id]
    snapshot = snapshots.get(wp.snapshot_id)
    if snapshot is None:
        return None
    cloud = snapshot.point_cloud
    odom_tform_cloud = get_a_tform_b(
        cloud.source.transforms_snapshot, ODOM_FRAME_NAME, cloud.source.frame_name_sensor
    )
    waypoint_tform_odom = SE3Pose.from_proto(wp.waypoint_tform_ko)
    waypoint_tform_cloud = _api_to_vtk_se3_pose(waypoint_tform_odom * odom_tform_cloud)

    pts = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
    poly = _numpy_to_poly_data(pts)

    z_arr = vtk.vtkFloatArray()
    z_arr.SetName("z_coord")
    for i in range(cloud.num_points):
        z_arr.InsertNextValue(pts[i, 2])
    poly.GetPointData().AddArray(z_arr)
    poly.GetPointData().SetActiveScalars("z_coord")

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(2)
    actor.SetUserTransform(waypoint_tform_cloud)
    return actor


def _create_waypoint_assembly(renderer, waypoints, snapshots, waypoint_id):
    assembly = vtk.vtkAssembly()

    axes = vtk.vtkAxesActor()
    axes.SetXAxisLabelText("")
    axes.SetYAxisLabelText("")
    axes.SetZAxisLabelText("")
    axes.SetTotalLength(0.2, 0.2, 0.2)
    assembly.AddPart(axes)

    try:
        pc_actor = _create_point_cloud_actor(waypoints, snapshots, waypoint_id)
        if pc_actor is not None:
            assembly.AddPart(pc_actor)
    except Exception as exc:
        print(f"[GraphNavVisualizer] Could not create point cloud for {waypoint_id}: {exc}")

    renderer.AddActor(assembly)
    return assembly


def _create_fiducial_actor(world_object, waypoint, renderer):
    fiducial = world_object.apriltag_properties
    odom_tform_fid = get_a_tform_b(
        world_object.transforms_snapshot,
        ODOM_FRAME_NAME,
        fiducial.frame_name_fiducial_filtered,
    )
    wp_tform_odom = SE3Pose.from_proto(waypoint.waypoint_tform_ko)
    wp_tform_fid = _api_to_vtk_se3_pose(wp_tform_odom * odom_tform_fid)

    plane = vtk.vtkPlaneSource()
    plane.SetCenter(0, 0, 0)
    plane.SetNormal(0, 0, 1)
    plane.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(plane.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.5, 0.7, 0.9)
    actor.SetScale(fiducial.dimensions.x, fiducial.dimensions.y, 1.0)
    renderer.AddActor(actor)
    return actor, wp_tform_fid


# ---------------------------------------------------------------------------
# Graph rendering (two modes: raw BFS layout vs anchored / seed-frame)
# ---------------------------------------------------------------------------

def _render_graph(graph, waypoints, snapshots, renderer, show_text):
    """BFS layout starting from the first waypoint. Returns average position."""
    wp_objects = {
        wp.id: _create_waypoint_assembly(renderer, waypoints, snapshots, wp.id)
        for wp in graph.waypoints
    }

    queue = [(graph.waypoints[0], np.eye(4))]
    visited = {}
    avg_pos = np.zeros(3)

    while queue:
        curr_wp, world_tform_curr = queue.pop(0)
        if curr_wp.id in visited:
            continue
        visited[curr_wp.id] = True

        wp_objects[curr_wp.id].SetUserTransform(_mat_to_vtk(world_tform_curr))
        if show_text:
            _make_text(curr_wp.annotations.name, world_tform_curr[:3, 3], renderer)

        # Fiducials in this waypoint's snapshot
        snapshot = snapshots.get(curr_wp.snapshot_id)
        if snapshot:
            for fid in snapshot.objects:
                if fid.HasField("apriltag_properties"):
                    actor, wp_tform_fid = _create_fiducial_actor(fid, curr_wp, renderer)
                    world_tform_fid = world_tform_curr @ _vtk_to_mat(wp_tform_fid)
                    actor.SetUserTransform(_mat_to_vtk(world_tform_fid))
                    if show_text:
                        _make_text(
                            str(fid.apriltag_properties.tag_id),
                            world_tform_fid[:3, 3],
                            renderer,
                        )

        for edge in graph.edges:
            if edge.id.from_waypoint == curr_wp.id and edge.id.to_waypoint not in visited:
                tform_to = SE3Pose.from_proto(edge.from_tform_to).to_matrix()
                world_tform_to = world_tform_curr @ tform_to
                _make_line(world_tform_curr[:3, 3], world_tform_to[:3, 3], renderer)
                queue.append((waypoints[edge.id.to_waypoint], world_tform_to))
                avg_pos += world_tform_to[:3, 3]
            elif edge.id.to_waypoint == curr_wp.id and edge.id.from_waypoint not in visited:
                tform_from = SE3Pose.from_proto(edge.from_tform_to).inverse().to_matrix()
                world_tform_from = world_tform_curr @ tform_from
                _make_line(world_tform_curr[:3, 3], world_tform_from[:3, 3], renderer)
                queue.append((waypoints[edge.id.from_waypoint], world_tform_from))
                avg_pos += world_tform_from[:3, 3]

    n = max(len(waypoints), 1)
    return avg_pos / n


def _render_anchored_graph(graph, waypoints, snapshots, anchors, anchored_world_objects,
                           renderer, show_text):
    """Render waypoints in seed frame using anchoring data. Returns average position."""
    avg_pos = np.zeros(3)
    count = 0

    for wp in graph.waypoints:
        if wp.id not in anchors:
            continue
        assembly = _create_waypoint_assembly(renderer, waypoints, snapshots, wp.id)
        seed_tform_wp = SE3Pose.from_proto(anchors[wp.id].seed_tform_waypoint).to_matrix()
        assembly.SetUserTransform(_mat_to_vtk(seed_tform_wp))
        if show_text:
            _make_text(wp.annotations.name, seed_tform_wp[:3, 3], renderer)
        avg_pos += seed_tform_wp[:3, 3]
        count += 1

    for edge in graph.edges:
        if edge.id.from_waypoint in anchors and edge.id.to_waypoint in anchors:
            seed_tform_from = SE3Pose.from_proto(
                anchors[edge.id.from_waypoint].seed_tform_waypoint
            ).to_matrix()
            from_tform_to = SE3Pose.from_proto(edge.from_tform_to).to_matrix()
            world_tform_to = seed_tform_from @ from_tform_to
            _make_line(seed_tform_from[:3, 3], world_tform_to[:3, 3], renderer)

    for awo in anchored_world_objects.values():
        if len(awo) < 3:
            continue
        actor, _ = _create_fiducial_actor(awo[2], awo[1], renderer)
        seed_tform_fid = SE3Pose.from_proto(awo[0].seed_tform_object).to_matrix()
        actor.SetUserTransform(_mat_to_vtk(seed_tform_fid))
        if show_text:
            _make_text(awo[0].id, seed_tform_fid[:3, 3], renderer)

    return avg_pos / max(count, 1)


# ---------------------------------------------------------------------------
# Public widget
# ---------------------------------------------------------------------------

class GraphNavWidget(QWidget):
    """
    A PyQt5 widget that renders a Boston Dynamics GraphNav map using VTK.

    Usage
    -----
        widget = GraphNavWidget(parent)
        widget.load_map("/path/to/map_folder", anchoring=False, show_waypoint_text=True)
    """

    def __init__(self, parent=None, placeholder_text=None):
        super().__init__(parent)

        self._vtk_widget = QVTKRenderWindowInteractor(self)

        # Dark background renderer
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.05, 0.1, 0.15)
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)

        # Terrain-style mouse interaction (pan / orbit / zoom)
        style = vtk.vtkInteractorStyleTerrain()
        self._vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._vtk_widget)

        # Placeholder text shown before any map is loaded
        text = "No map loaded.\nSelect a GraphNav folder and click Load."
        if placeholder_text is not None:
            text = placeholder_text
    
        self._placeholder = QLabel(text)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self._placeholder)

        self._placeholder.setVisible(True)
        self._vtk_widget.setVisible(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_map(self, path: str, *, anchoring: bool = False, show_waypoint_text: bool = True,
                 show_world_object_text: bool = True):
        """
        Load and render the GraphNav map at *path*.

        Parameters
        ----------
        path:
            Root directory of the map (contains a ``graph`` file and
            ``waypoint_snapshots/`` / ``edge_snapshots/`` sub-directories).
        anchoring:
            If True, render in seed frame using anchoring data.
        show_waypoint_text:
            Show waypoint name labels in the 3-D view.
        show_world_object_text:
            Show AprilTag / world-object labels in the 3-D view.
        """
        try:
            graph, waypoints, snapshots, _, anchors, anchored_wos = _load_map(path)
        except Exception as exc:
            print(f"[GraphNavWidget] Failed to load map: {exc}")
            self._placeholder.setText(f"Failed to load map:\n{exc}")
            return

        # Clear previous scene
        self._renderer.RemoveAllViewProps()

        if anchoring:
            if not anchors:
                print("[GraphNavWidget] No anchors found; falling back to BFS layout.")
                avg_pos = _render_graph(graph, waypoints, snapshots, self._renderer,
                                        show_waypoint_text)
            else:
                avg_pos = _render_anchored_graph(graph, waypoints, snapshots, anchors,
                                                 anchored_wos, self._renderer,
                                                 show_waypoint_text)
        else:
            avg_pos = _render_graph(graph, waypoints, snapshots, self._renderer,
                                    show_waypoint_text)

        # Position the camera above the centroid, looking down
        cam_pos = avg_pos + np.array([-1.0, 0.0, 5.0])
        camera = self._renderer.GetActiveCamera()
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition(*cam_pos)
        camera.SetFocalPoint(*avg_pos)
        self._renderer.ResetCamera()

        # Switch from placeholder to VTK view
        self._placeholder.setVisible(False)
        self._vtk_widget.setVisible(True)
        self._vtk_widget.Initialize()
        self._vtk_widget.Start()
        self._vtk_widget.GetRenderWindow().Render()

    def clear(self):
        """Remove all actors and return to the placeholder state."""
        self._renderer.RemoveAllViewProps()
        self._vtk_widget.GetRenderWindow().Render()
        self._vtk_widget.setVisible(False)
        self._placeholder.setText("No map loaded.\nSelect a GraphNav folder and click Load.")
        self._placeholder.setVisible(True)

    def refresh(self, graph, waypoints: dict, snapshots: dict, *, show_waypoint_text: bool = True):
        """
        Re-render from live graph data (e.g. polled from the robot during recording).
        Accepts the same data structures that _render_graph expects.
        """
        if not graph or not graph.waypoints:
            return

        self._renderer.RemoveAllViewProps()
        avg_pos = _render_graph(graph, waypoints, snapshots, self._renderer, show_waypoint_text)

        camera = self._renderer.GetActiveCamera()
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition(*(avg_pos + np.array([-1.0, 0.0, 5.0])))
        camera.SetFocalPoint(*avg_pos)
        self._renderer.ResetCamera()

        self._placeholder.setVisible(False)
        self._vtk_widget.setVisible(True)
        self._vtk_widget.GetRenderWindow().Render()


# ---------------------------------------------------------------------------
# Map loader (pure data, no VTK / Qt)
# ---------------------------------------------------------------------------

def _load_map(path):
    """
    Parse a GraphNav map directory and return its data structures.

    Returns
    -------
    (graph, waypoints, waypoint_snapshots, edge_snapshots, anchors, anchored_world_objects)
    """
    with open(os.path.join(path, "graph"), "rb") as f:
        data = f.read()

    graph = map_pb2.Graph()
    graph.ParseFromString(data)

    waypoints = {}
    snapshots = {}
    edge_snapshots = {}
    anchors = {}
    anchored_wos = {}

    # Index anchored world objects (placeholder tuples filled in below)
    for awo in graph.anchoring.objects:
        anchored_wos[awo.id] = (awo,)

    for wp in graph.waypoints:
        waypoints[wp.id] = wp
        if not wp.snapshot_id:
            continue
        snap_file = os.path.join(path, "waypoint_snapshots", wp.snapshot_id)
        if not os.path.exists(snap_file):
            continue
        with open(snap_file, "rb") as f:
            snap = map_pb2.WaypointSnapshot()
            try:
                snap.ParseFromString(f.read())
                snapshots[snap.id] = snap
            except Exception as exc:
                print(f"[_load_map] Could not parse snapshot {snap_file}: {exc}")
                continue

            for fid in snap.objects:
                if not fid.HasField("apriltag_properties"):
                    continue
                str_id = str(fid.apriltag_properties.tag_id)
                if str_id in anchored_wos and len(anchored_wos[str_id]) == 1:
                    anchored_wos[str_id] = (anchored_wos[str_id][0], wp, fid)

    for edge in graph.edges:
        if not edge.snapshot_id:
            continue
        snap_file = os.path.join(path, "edge_snapshots", edge.snapshot_id)
        if not os.path.exists(snap_file):
            continue
        with open(snap_file, "rb") as f:
            try:
                snap = map_pb2.EdgeSnapshot()
                snap.ParseFromString(f.read())
                edge_snapshots[snap.id] = snap
            except Exception as e:
                # TODO: Fix error: Error parsing message with type 'bosdyn.api.graph_nav.EdgeSnapshot'
                # Only appears on Windows when trieng to load the map.
                # Works fine when ignoring Exception
                # print(e)
                pass

    for anchor in graph.anchoring.anchors:
        anchors[anchor.id] = anchor

    print(
        f"[_load_map] {len(graph.waypoints)} waypoints, {len(graph.edges)} edges, "
        f"{len(anchors)} anchors, {len(anchored_wos)} anchored world objects"
    )
    return graph, waypoints, snapshots, edge_snapshots, anchors, anchored_wos