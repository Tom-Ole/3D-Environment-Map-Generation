# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface integrating options to record maps with WASD controls. """
import logging
import datetime
import os
import time

logger = logging.getLogger(__name__)


import utils.route.graph_nav_util as graph_nav_util

from bosdyn.api.graph_nav import map_pb2, map_processing_pb2, recording_pb2
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.recording import GraphNavRecordingServiceClient, NotReadyYetError



class RecordingInterface(object):
    """Recording service command line interface."""

    def __init__(self, robot, download_filepath, client_metadata, recording_client, graph_nav_client, map_processing_client):
        # Keep the robot instance and it's ID.
        self._robot = robot

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Filepath for the location to put the downloaded graph and snapshots.
        # TODO: maybe change to just download_filepath
        self._download_filepath = os.path.join(download_filepath, "route")

        # Set up the recording service client.
        self._recording_client = recording_client

        # Create the recording environment.
        self._recording_environment = GraphNavRecordingServiceClient.make_recording_environment(
            waypoint_env=GraphNavRecordingServiceClient.make_waypoint_environment(
                client_metadata=client_metadata))

        # Set up the graph nav service client.
        self._graph_nav_client = graph_nav_client

        self._map_processing_client = map_processing_client

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()


    def should_we_start_recording(self):
        # Before starting to record, check the state of the GraphNav system.
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            # Check that the graph has waypoints. If it does, then we need to be localized to the graph
            # before starting to record
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    # Not localized to anything in the map. The best option is to clear the graph or
                    # attempt to localize to the current map.
                    # Returning false since the GraphNav system is not in the state it should be to
                    # begin recording.
                    return False
        # If there is no graph or there exists a graph that we are localized to, then it is fine to
        # start recording, so we return True.
        return True

    def _clear_map(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def _start_recording(self, *args):
        """Start recording a map."""
        if not self.should_we_start_recording():
            raise RuntimeError(
                "Cannot start recording: localize to the map or clear the graph first."
            )
        try:
            self._recording_client.start_recording(
                recording_environment=self._recording_environment)
            logger.info("Successfully started recording a map.")
        except Exception as err:
            logger.error("Start recording failed: %s", err)
            raise

    def _stop_recording(self, *args):
        """Stop or pause recording a map."""
        first_iter = True
        while True:
            try:
                self._recording_client.stop_recording()
                logger.info("Successfully stopped recording a map.")
                break
            except NotReadyYetError:
                if first_iter:
                    logger.info("Cleaning up recording...")
                first_iter = False
                time.sleep(1.0)
                continue
            except Exception as err:
                logger.error("Stop recording failed: %s", err)
                raise

    def _get_recording_status(self, *args):
        """Get the recording service's status."""
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print('The recording service is on.')
        else:
            print('The recording service is off.')

    def _create_default_waypoint(self, *args):
        """Create a default waypoint at the robot's current location."""
        resp = self._recording_client.create_waypoint(waypoint_name='default')
        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            logger.info("Successfully created a waypoint.")
        else:
            raise RuntimeError(f"Could not create a waypoint (status={resp.status}).")

    def _download_full_graph(self, *args):
        """Download the graph and snapshots from the robot."""
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            raise RuntimeError("Failed to download the graph from the robot.")
        self._write_full_graph(graph)
        logger.info(
            "Graph downloaded with %d waypoints and %d edges",
            len(graph.waypoints),
            len(graph.edges),
        )
        # Download the waypoint and edge snapshots.
        self._download_and_write_waypoint_snapshots(graph.waypoints)
        self._download_and_write_edge_snapshots(graph.edges)

    def _write_full_graph(self, graph):
        """Download the graph from robot to the specified, local filepath location."""
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, 'graph', graph_bytes)

    def _download_and_write_waypoint_snapshots(self, waypoints):
        """Download the waypoint snapshots from robot to the specified, local filepath location."""
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(
                    waypoint.snapshot_id)
            except Exception as e:
                logger.warning(f'Failed to download waypoint snapshot {waypoint.snapshot_id}: {e}')
                continue
            self._write_bytes(os.path.join(self._download_filepath, 'waypoint_snapshots'),
                              str(waypoint.snapshot_id), waypoint_snapshot.SerializeToString())
            num_waypoint_snapshots_downloaded += 1
            print(
                f'Downloaded {num_waypoint_snapshots_downloaded} of the total {len(waypoints)} waypoint snapshots.'
            )

    def _download_and_write_edge_snapshots(self, edges):
        """Download the edge snapshots from robot to the specified, local filepath location."""
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception as e:
                logger.warning(f'Failed to download edge snapshot {edge.snapshot_id}: {e}')
                continue
            self._write_bytes(os.path.join(self._download_filepath, 'edge_snapshots'),
                              str(edge.snapshot_id), edge_snapshot.SerializeToString())
            num_edge_snapshots_downloaded += 1
            print(
                f'Downloaded {num_edge_snapshots_downloaded} of the total {num_to_download} edge snapshots.'
            )

    def _write_bytes(self, filepath, filename, data):
        """Write data to a file."""
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, filename), 'wb+') as f:
            f.write(data)
            f.close()

    def _update_graph_waypoint_and_edge_ids(self, do_print=False):
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Empty graph.')
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id, do_print)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""
        self._update_graph_waypoint_and_edge_ids(do_print=True)

    def _create_new_edge(self, *args):
        """Create new edge between existing waypoints in map."""

        if len(args[0]) != 2:
            print('ERROR: Specify the two waypoints to connect (short code or annotation).')
            return

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        from_id = graph_nav_util.find_unique_waypoint_id(args[0][0], self._current_graph,
                                                         self._current_annotation_name_to_wp_id)
        to_id = graph_nav_util.find_unique_waypoint_id(args[0][1], self._current_graph,
                                                       self._current_annotation_name_to_wp_id)

        print(f'Creating edge from {from_id} to {to_id}.')

        from_wp = self._get_waypoint(from_id)
        if from_wp is None:
            return

        to_wp = self._get_waypoint(to_id)
        if to_wp is None:
            return

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_id
        new_edge.id.to_waypoint = to_id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        print(f'edge transform = {new_edge.from_tform_to}')

        # Send request to add edge to map
        self._recording_client.create_edge(edge=new_edge)

    def _create_loop(self, *args):
        """Create edge from last waypoint to first waypoint."""

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        if len(self._current_graph.waypoints) < 2:
            raise RuntimeError(
                f"Graph contains {len(self._current_graph.waypoints)} waypoints; "
                "at least two are needed to create a loop."
            )

        sorted_waypoints = graph_nav_util.sort_waypoints_chrono(self._current_graph)
        edge_waypoints = [sorted_waypoints[-1][0], sorted_waypoints[0][0]]

        self._create_new_edge(edge_waypoints)


    def _optimize_anchoring(self, *args):
        """Call anchoring optimization on the server, producing a globally optimal reference frame for waypoints to be expressed in."""
        response = self._map_processing_client.process_anchoring(
            params=map_processing_pb2.ProcessAnchoringRequest.Params(),
            modify_anchoring_on_server=True, stream_intermediate_results=False,
            apply_gps_results=False)
        if response.status == map_processing_pb2.ProcessAnchoringResponse.STATUS_OK:
            print(f'Optimized anchoring after {response.iteration} iteration(s).')
            # If we are using GPS, the GPS coordinates in the graph have been changed, so we need
            # to re-download the graph.
            if False:
                print('Downloading updated graph...')
                self._current_graph = self._graph_nav_client.download_graph()
        else:
            print(f'Error optimizing {response}')

    def _get_waypoint(self, id):
        """Get waypoint from graph (return None if waypoint not found)"""

        if self._current_graph is None:
            self._current_graph = self._graph_nav_client.download_graph()

        for waypoint in self._current_graph.waypoints:
            if waypoint.id == id:
                return waypoint

        print(f'ERROR: Waypoint {id} not found in graph.')
        return None

    def _get_transform(self, from_wp, to_wp):
        """Get transform from from-waypoint to to-waypoint."""

        from_se3 = from_wp.waypoint_tform_ko
        from_tf = SE3Pose(
            from_se3.position.x, from_se3.position.y, from_se3.position.z,
            Quat(w=from_se3.rotation.w, x=from_se3.rotation.x, y=from_se3.rotation.y,
                 z=from_se3.rotation.z))

        to_se3 = to_wp.waypoint_tform_ko
        to_tf = SE3Pose(
            to_se3.position.x, to_se3.position.y, to_se3.position.z,
            Quat(w=to_se3.rotation.w, x=to_se3.rotation.x, y=to_se3.rotation.y,
                 z=to_se3.rotation.z))

        from_T_to = from_tf.mult(to_tf.inverse())
        return from_T_to.to_proto()

    def start(self):
        self._clear_map()
        self._start_recording()

    def create_waypoint(self):
        self._create_default_waypoint()

    def stop(self, create_loop: bool = False):
        if create_loop:
            self._create_loop()


        self._stop_recording()
        self._optimize_anchoring()
        self._download_full_graph()



