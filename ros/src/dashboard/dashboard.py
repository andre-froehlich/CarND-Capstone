#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
import numpy as np
import pygame
import rospy
from ast import literal_eval as make_tuple
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Bool

# couple of colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


class Dashboard(object):
    # pygame display
    _screen = None
    # actual window dimensions
    _window_dimensions = None
    # arbitrary dimensions to fit track
    _screen_dimensions = None

    _background_color = None

    _current_pose = None
    _base_waypoints = None
    _len_waypoints = -1
    _dbw_enabled = False
    _current_velocity = None
    _twist_cmd = None

    # shows the track in white on black background
    _track_image = None
    # is updated each iteration with new values
    _dashboard_img = None

    def __init__(self):
        rospy.init_node('dashboard_node')

        # read window dimensions and background color from dashboard.launch
        self._window_dimensions = make_tuple(rospy.get_param('~window_dimensions', (300, 200)))
        self._background_color = make_tuple(rospy.get_param('~background_color', BLACK))
        # calculate screen dimensions from window dimensions
        self._screen_dimensions = (self._window_dimensions[0] * 3, self._window_dimensions[1] * 3)

        # subscribe to pose and waypoints
        rospy.Subscriber('/current_pose', PoseStamped, self._set_current_pose)
        rospy.Subscriber('/base_waypoints', Lane, self._set_base_waypoints)

        # subscribe to traffic light and obstacle topics
        # rospy.Subscriber('/traffic_waypoint', Lane, self._set_traffic_waypoints)
        # rospy.Subscriber('/obstacle_waypoint', Lane, self._set_obstacle_waypoints)

        # is Drive-By-Wire enabled?
        rospy.Subscriber('/dbw_enabled', Bool, self._set_dbw_enabled)

        # current velocity and twist topic
        rospy.Subscriber('/current_velocity', TwistStamped, self._set_current_velocity)
        rospy.Subscriber('/twist_cmd', TwistStamped, self._set_twist_cmd)

        # initialize screen with given parameters from dashboard.launch
        self._init_screen()

        # start loop
        self._loop()

    def _init_screen(self):
        # produces error msgs and works without
        # pygame.init()
        # set caption of pygame window
        pygame.display.set_caption("Happy Robots Dashboard")
        # create screen wof window dimensions set in the dashboard.launch file
        self._screen = pygame.display.set_mode(self._window_dimensions, pygame.DOUBLEBUF)
        # set background color from dashboard.launch file
        self._screen.fill(self._background_color)

    def _update_screen(self):
        if self._dashboard_img is not None:
            # if dashboard_img is set, resize it to window dimension, generate bytes and draw it with pygame
            self._dashboard_img = pygame.image.fromstring(
                cv2.resize(self._dashboard_img, self._window_dimensions).tobytes(), self._window_dimensions, 'RGB')
            # put on _screen
            self._screen.blit(self._dashboard_img, (0, 0))
            # update pygame screen
            pygame.display.flip()

    def _draw_track(self):
        # transform base waypoints to vertices for cv2.polylines
        xs = list()
        ys = list()
        for wp in self._base_waypoints.waypoints:
            xs.append(wp.pose.pose.position.x)
            #  normalize y values
            ys.append(self._screen_dimensions[1] - (wp.pose.pose.position.y - 1000.))
        vertices = [np.column_stack((xs, ys)).astype(np.int32)]

        # create empty image with screen dimensions
        self._track_image = np.zeros((self._screen_dimensions[0], self._screen_dimensions[1], 3), dtype=np.uint8)
        # draw polylines of the track
        cv2.polylines(self._track_image, vertices, False, WHITE, 5)

        # draw initial car position
        self._current_pose = self._base_waypoints.waypoints[-1].pose.pose
        self._draw_current_position()

    def _draw_current_position(self):
        if self._current_pose is not None:
            x = int(self._current_pose.position.x)
            y = int(self._screen_dimensions[1] - (self._current_pose.position.y - 1000.))
            cv2.circle(self._dashboard_img, (x, y), 10, RED, -1)

    def _draw_traffic_lights(self):
        # if self._traffic_lights is not None:
        # TODO: implement

        pass

    def _loop(self):
        # 1Hz should be enough
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self._base_waypoints is not None:
                # get copy of track_image
                self._dashboard_img = np.copy(self._track_image)

                self._draw_current_position()

                # test text
                header = "Dashboard"
                cv2.putText(self._dashboard_img, header, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, WHITE, 2)

                # update screen with new image and refresh window
                self._update_screen()

            # wait for next iteration
            rate.sleep()

        # clean shutdown
        self.close()

    def close(self):
        rospy.logwarn("close")
        pygame.quit()

    def _set_current_pose(self, msg):
        self._current_pose = msg.pose
        rospy.loginfo(
            "Received new position: x={}, y={}".format(self._current_pose.position.x, self._current_pose.position.y))

    def _set_base_waypoints(self, waypoints):
        if self._base_waypoints is None:
            self._base_waypoints = waypoints
            self._len_waypoints = len(self._base_waypoints.waypoints)
            rospy.logwarn("Waypoints loaded... found {}.".format(self._len_waypoints))
            # draws track image right after setting waypoints
            # this way it only has to be done once
            self._draw_track()

    def _set_traffic_waypoints(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def _set_obstacle_waypoints(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def _set_dbw_enabled(self, msg):
        self._dbw_enabled = msg

    def _set_current_velocity(self, msg):
        self._current_velocity = msg

    def _set_twist_cmd(self, msg):
        self._twist_cmd = msg


if __name__ == '__main__':
    Dashboard()
