# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface for graph nav with options to download/upload a map
and to navigate a map."""

import argparse
import logging
import math
import os
import sys
import time
import traceback

import google.protobuf.timestamp_pb2
import utils.route.graph_nav_util as graph_nav_util
import grpc

from utils.image.get_images import get_image
from utils.image.ImageOptions import ImageOptions, ImageSources
from utils.image.colmap_writer import ColmapWriter
from pathlib import Path

import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.api import geometry_pb2, power_pb2, robot_state_pb2
from bosdyn.api.gps import gps_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on_motors, safe_power_off_motors
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

logger = logging.getLogger(__name__)


class GraphNavInterface(object):
    """GraphNav service command line interface."""


    def __init__(self, controller, output_path, lease,
                 capture_interval_m: float = 0.1):
        
        self.controller = controller

        self._robot = self.controller.robot

        self._lease = lease

        self._robot.time_sync.wait_for_sync()

        # Clients
        self._robot_command_client = self.controller.command_client
        self._robot_state_client = self.controller.robot_state_client
        self._graph_nav_client = self.controller.graph_nav_client
        self._power_client = self.controller.power_client

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Filepath for uploading a saved graph's and snapshots too.
        self.output_path = output_path
        self._upload_filepath = output_path
        self._capture_interval_m = capture_interval_m


    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)



    def _clear_graph_and_cache(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and
        edges.

        Also clears the disk cache.
        """
        return self._graph_nav_client.clear_graph_and_cache()

    # @do_not_publish_end

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print('No waypoint specified to initialize to.')
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.1,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)


    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print('Loading the graph from disk into local storage...')
        with open(self._upload_filepath + '/graph', 'rb') as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print(
                f'Loaded graph has {len(self._current_graph.waypoints)} waypoints and {len(self._current_graph.edges)} edges'
            )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(f'{self._upload_filepath}/waypoint_snapshots/{waypoint.snapshot_id}',
                      'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(f'{self._upload_filepath}/edge_snapshots/{edge.snapshot_id}',
                      'rb') as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print('Uploading the graph and snapshots to the robot...')
        time_before = time.time()
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload any missing snapshots to the robot.
        upload_individually = False
        try:
            self._graph_nav_client.upload_snapshots(
                graph_nav_pb2.UploadSnapshotsRequest.Snapshots(waypoint_snapshots=[],
                                                               edge_snapshots=[]))
        except Exception as e:
            logger.warning("Empty UploadSnapshots request failed, falling back to individual uploads: %s", e)
            upload_individually = True

        if upload_individually:
            for snapshot_id in response.unknown_waypoint_snapshot_ids:
                waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
                self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
                print(f'Uploaded {waypoint_snapshot.id}')
            for snapshot_id in response.unknown_edge_snapshot_ids:
                edge_snapshot = self._current_edge_snapshots[snapshot_id]
                self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
                print(f'Uploaded {edge_snapshot.id}')
        else:
            # Upload in groups of 16MB.
            kMaxBytes = 16 * 1024 * 1024
            snapshots = []
            num_bytes = 0

            # Upload waypoint snapshots.
            for snapshot_id in response.unknown_waypoint_snapshot_ids:
                this_bytes = self._current_waypoint_snapshots[snapshot_id].ByteSize()
                if len(snapshots) > 0 and this_bytes + num_bytes > kMaxBytes:
                    print(f'Uploading {len(snapshots)} waypoint snapshots')
                    self._graph_nav_client.upload_snapshots(
                        graph_nav_pb2.UploadSnapshotsRequest.Snapshots(
                            waypoint_snapshots=snapshots, edge_snapshots=[]))
                    snapshots = []
                    num_bytes = 0
                snapshots.append(self._current_waypoint_snapshots[snapshot_id])
                num_bytes += this_bytes
            if len(snapshots) > 0:
                print(f'Uploading final {len(snapshots)} waypoint snapshots')
                self._graph_nav_client.upload_snapshots(
                    graph_nav_pb2.UploadSnapshotsRequest.Snapshots(waypoint_snapshots=snapshots,
                                                                   edge_snapshots=[]))

            # Upload edge snapshots.
            snapshots = []
            num_bytes = 0
            for snapshot_id in response.unknown_edge_snapshot_ids:
                this_bytes = self._current_edge_snapshots[snapshot_id].ByteSize()
                if len(snapshots) > 0 and this_bytes + num_bytes > kMaxBytes:
                    print(f'Uploading {len(snapshots)} edge snapshots')
                    self._graph_nav_client.upload_snapshots(
                        graph_nav_pb2.UploadSnapshotsRequest.Snapshots(
                            waypoint_snapshots=[], edge_snapshots=snapshots))
                    snapshots = []
                    num_bytes = 0
                snapshots.append(self._current_edge_snapshots[snapshot_id])
                num_bytes += this_bytes
            if len(snapshots) > 0:
                print(f'Uploading final {len(snapshots)} edge snapshots')
                self._graph_nav_client.upload_snapshots(
                    graph_nav_pb2.UploadSnapshotsRequest.Snapshots(waypoint_snapshots=[],
                                                                   edge_snapshots=snapshots))
        upload_time = time.time() - time_before
        print(
            f'Uploaded graph and {len(response.unknown_waypoint_snapshot_ids)} (of {len(self._current_graph.waypoints)}) waypoints and {len(response.unknown_edge_snapshot_ids)} (of {len(self._current_graph.edges)}) edges, elapsed time {round(upload_time * 1000)}ms'
        )

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print('\n')
            print(
                'Upload complete! The robot is currently not localized to the map; please localize'
                ' the robot using commands (2) or (3) before attempting a navigation command.')



    def _my_navigate_route(self, *args):
        """Navigate waypoints in recording order; capture at each waypoint by distance interval."""

        interval_m: float = self._capture_interval_m

        if not self._current_graph or not self._current_graph.waypoints:
            raise RuntimeError("No graph loaded or graph is empty.")

        loc_state = self._graph_nav_client.get_localization_state()
        if not loc_state.localization.waypoint_id:
            raise RuntimeError("Robot is not localized to the map.")

        ordered_waypoints = graph_nav_util.sort_waypoints_chrono(self._current_graph)
        ordered_waypoint_ids = [wp[0] for wp in ordered_waypoints]
        logger.info(
            "Built route of %d waypoints in recording order (capture interval=%.2fm).",
            len(ordered_waypoint_ids),
            interval_m,
        )

        #  Setup image capture 
        robot_image_client = self._robot.ensure_client('image')
        frame_id = 0

        if not self.toggle_power(should_power_on=True):
            raise RuntimeError("Failed to power on the robot.")

        logger.info("Capturing initial image (frame %05d)...", frame_id)
        self.controller.get_image(frame_id, self._lease)
        frame_id += 1

        loc = self._graph_nav_client.get_localization_state().localization
        last_capture_pos = loc.seed_tform_body.position

        # velocity_limit = geometry_pb2.SE2VelocityLimit(
        # max_vel=geometry_pb2.SE2Velocity(
        #     linear=geometry_pb2.Vec2(x=self.NAVIGATE_SPEED_MS, y=1e6),
        #     angular=1e6,
        # ),
        # min_vel=geometry_pb2.SE2Velocity(
        #     linear=geometry_pb2.Vec2(x=-1e6, y=-1e6),
        #     angular=-1e6,
        #     ),
        # )

        # travel_params = graph_nav_pb2.TravelParams(velocity_limit=velocity_limit)

        #  Navigate waypoint by waypoint in recording order 
        for i, waypoint_id in enumerate(ordered_waypoint_ids):
            logger.info(
                "Navigating to waypoint %d/%d: %s",
                i + 1,
                len(ordered_waypoint_ids),
                waypoint_id,
            )

            nav_to_cmd_id = None
            is_finished = False

            while not is_finished:
                try:
                    nav_to_cmd_id = self._graph_nav_client.navigate_to(
                        waypoint_id, 1.0, 
                        command_id=nav_to_cmd_id, 
                        #travel_params=travel_params
                        )
                except ResponseError as e:
                    logger.error("Error while navigating to %s: %s", waypoint_id, e)
                    raise RuntimeError(f"Navigation failed at waypoint {waypoint_id}: {e}") from e

                time.sleep(0.5)
                is_finished = self._check_success(nav_to_cmd_id)

            #  Robot has stopped; check distance from last capture 
            loc = self._graph_nav_client.get_localization_state().localization
            current_pos = loc.seed_tform_body.position

            dist = math.sqrt(
                (current_pos.x - last_capture_pos.x) ** 2 +
                (current_pos.y - last_capture_pos.y) ** 2
            )

            should_capture = interval_m == 0.0 or dist >= interval_m
            if should_capture:
                logger.info(
                    "Capturing at waypoint %s (%.2fm from last, frame %05d).",
                    waypoint_id,
                    dist,
                    frame_id,
                )
                self.controller.get_image(frame_id, self._lease)
                frame_id += 1
                last_capture_pos = current_pos
            else:
                logger.info(
                    "Skipping capture at waypoint %s (%.2fm from last, interval %.2fm).",
                    waypoint_id,
                    dist,
                    interval_m,
                )

        logger.info("Route complete. Captured %d image(s) total.", frame_id)

        if self._powered_on and not self._started_powered_on:
            self.toggle_power(should_power_on=False)




    def clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and
        edges."""
        return self._graph_nav_client.clear_graph()

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on_motors(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10)  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off_motors(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit
        when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            raise RuntimeError("Robot got lost when navigating the route.")
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            raise RuntimeError("Robot got stuck when navigating the route.")
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            raise RuntimeError("Robot is impaired and cannot continue navigation.")
        else:
            # Navigation command is not complete yet.
            return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(RobotCommandBuilder.safe_power_off_command(),
                                                     end_time_secs=time.time())

    def run(self):
        """Upload the graph, localize the robot, and navigate all waypoints while capturing images."""
        self._upload_graph_and_snapshots()

        # stand up
        # TODO: stand up

        try:
            self._set_initial_localization_fiducial()
        except Exception as e:
            logger.warning("Fiducial localization failed, falling back to waypoint localization: %s", e)
            self._set_initial_localization_waypoint([self._current_graph.waypoints[0].id])

        self._my_navigate_route()




