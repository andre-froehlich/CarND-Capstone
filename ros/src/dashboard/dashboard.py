#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab
import pygame
from pygame.locals import *

import rospy
from ast import literal_eval as make_tuple
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Bool


class Dashboard(object):
    _screen = None
    _window_dimensions = None
    _background_color = None

    _current_pose = None
    _base_waypoints = None
    _len_waypoints = -1
    _dbw_enabled = False
    _current_velocity = None
    _twist_cmd = None

    def __init__(self):
        rospy.init_node('dashboard_node')

        _window_dimensions = make_tuple(rospy.get_param('~window_dimensions', (300, 200)))
        _background_color = make_tuple(rospy.get_param('~background_color', (255, 255, 255)))

        rospy.Subscriber('/current_pose', PoseStamped, self._set_current_pose)
        rospy.Subscriber('/base_waypoints', Lane, self._set_base_waypoints)
        # rospy.Subscriber('/traffic_waypoint', Lane, self._set_traffic_waypoints)
        # rospy.Subscriber('/obstacle_waypoint', Lane, self._set_obstacle_waypoints)

        rospy.Subscriber('/dbw_enabled', Bool, self._set_dbw_enabled)
        rospy.Subscriber('/current_velocity', TwistStamped, self._set_current_velocity)
        rospy.Subscriber('/twist_cmd', TwistStamped, self._set_twist_cmd)

        # TODO
        pygame.display.set_caption("Dashboard")
        self._screen = pygame.display.set_mode(_window_dimensions)
        self._screen.fill(_background_color)

        self._loop()

    def _set_current_pose(self, msg):
        # TODO: Implement
        self._current_pose = msg.pose
        rospy.loginfo(
            "Received new position: x={}, y={}".format(self._current_pose.position.x, self._current_pose.position.y))

    def _set_base_waypoints(self, waypoints):
        if self._base_waypoints is None:
            self._base_waypoints = waypoints
            self._len_waypoints = len(self._base_waypoints.waypoints)
            rospy.loginfo("Waypoints loaded... found {}.".format(self._len_waypoints))

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

    def _print_track(self):
        # TODO: implement
        # figure with size of 800x600 (6inch * 100dpi = 600)
        # fig = pylab.figure(figsize=[8, 6], dpi=100)
        # ax = fig.gca()
        # ax.plot([1, 2, 4])
        # canvas = agg.FigureCanvasAgg(fig)
        # canvas.draw()
        # renderer = canvas.get_renderer()
        # raw_data = renderer.tostring_rgb()
        #
        # size = canvas.get_width_height()
        # surf = pygame.image.fromstring(raw_data, size, "RGB")
        # self._screen.blit(surf, (0, 0))
        
        pygame.display.flip()

    def _loop(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self._base_waypoints is not None:
                self._print_track()

            rate.sleep()

        self.close()

    def close(self):
        rospy.logwarn("close")
        pygame.quit()


if __name__ == '__main__':
    Dashboard()
