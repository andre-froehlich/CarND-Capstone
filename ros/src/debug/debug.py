#!/usr/bin/env python

import numpy as np

import yaml
from ast import literal_eval as make_tuple
from cv_bridge import CvBridge
from math import sqrt, pi, tan
import cv2
import matplotlib.pyplot as plt

import pygame
import rospy
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from styx_msgs.msg import Lane, TrafficLightArray, Debug
from utilities import utils

# couple of colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)

TRAFFIC_STATES = {0: RED, 1: YELLOW, 2: GREEN}


class Dashboard(object):
    # pygame display
    _screen = None
    # actual window dimensions
    _window_dimensions = None
    # arbitrary dimensions to fit track
    _screen_dimensions = None

    _background_color = None

    _debug_img = None

    _image = None
    _bridge = CvBridge()

    _debug_msg = None

    def __init__(self):
        rospy.init_node('debug_node')

        self._read_params()

        # calculate screen dimensions from window dimensions
        self._screen_dimensions = (self._window_dimensions[0] * 3, self._window_dimensions[1] * 3)

        rospy.Subscriber('/debug_msg', Debug, self.set_debug_msg)

        # initialize screen with given parameters from debug.launch
        self._screen = self._init_screen()

        # start loop
        self._loop()

    def _read_params(self):
        # read window dimensions and background color from debug.launch
        self._window_dimensions = make_tuple(rospy.get_param('~window_dimensions', (300, 200)))
        self._background_color = make_tuple(rospy.get_param('~background_color', str(BLACK)))
        self._text_color = make_tuple(rospy.get_param('~text_color', str(WHITE)))
        self._text_shadow_color = make_tuple(rospy.get_param('~text_shadow_color', str(GREY)))
        self._ego_color = make_tuple(rospy.get_param('~ego_color', str(MAGENTA)))
        self._final_waypoints_color = make_tuple(rospy.get_param('~final_waypoints_color', str(GREEN)))
        self._base_waypoints_color = make_tuple(rospy.get_param('~base_waypoints_color', str(WHITE)))

    def _init_screen(self):
        # produces error msgs and works without
        # pygame.init()
        # set caption of pygame window
        pygame.display.set_caption("Happy Robots Debug")
        # create screen wof window dimensions set in the debug.launch file
        screen = pygame.display.set_mode(self._window_dimensions, pygame.DOUBLEBUF)
        # set background color from debug.launch file
        screen.fill(self._background_color)

        return screen

    def _update_screen(self, screen, img):
        if screen is not None and img is not None:
            # if dashboard_img is set, resize it to window dimension, generate bytes and draw it with pygame
            image = pygame.image.fromstring(cv2.resize(img, self._window_dimensions).tobytes(), self._window_dimensions, 'RGB')
            # put on _screen
            screen.blit(image, (0, 0))
            # update pygame screen
            pygame.display.flip()

    def _loop(self):
        # 1Hz is not enough better make it 5
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self._debug_msg is not None:
                v_ref = self._debug_msg.v_ref
                v_cur = self._debug_msg.v_cur
                brake = self._debug_msg.brake
                throttle = self._debug_msg.throttle
                v_err = self._debug_msg.v_err
                lowpass_out = self._debug_msg.lowpass_out
                steer = self._debug_msg.steer
                twist_linear_x = self._debug_msg.twist_linear_x
                twist_angular_z = self._debug_msg.twist_angular_z

                if len(v_ref) > 50:
                    self._debug_img = np.empty((self._screen_dimensions[0], self._screen_dimensions[1], 3), dtype=np.uint8)
                    fig, axarr = plt.subplots(3, sharex=True)
                    p1 = axarr[0].plot(v_err, color='r', label='Lowpass Input')
                    p2 = axarr[0].plot(lowpass_out, color='g', label='Lowpass Output')
                    axarr[0].set_ylabel('v_err(r)\nOutput(g)', color='b')
                    for tl in axarr[0].get_yticklabels():
                        tl.set_color('b')
                    ps = p1 + p2
                    lps = [l.get_label() for l in ps]
                    plt.legend(ps, lps, loc=2)

                    p3 = axarr[1].plot(v_ref, color='r', label='V_ref')
                    p4 = axarr[1].plot(v_cur, color='g', label='V_cur')
                    axarr[1].set_ylabel('V_ref(r)\nV_cur(g)', color='b')
                    for tl in axarr[1].get_yticklabels():
                        tl.set_color('b')
                    ps = p3 + p4
                    lps = [l.get_label() for l in ps]
                    plt.legend(ps, lps, loc=2)

                    p5 = axarr[2].plot(brake, color='r', label='Brake')
                    p6 = axarr[2].plot(throttle, color='g', label='Throttle')
                    axarr[2].set_ylabel('Brake(r)\nThrottle(g)', color='b')
                    for tl in axarr[2].get_yticklabels():
                        tl.set_color('b')
                    ps = p5 + p6
                    lps = [l.get_label() for l in ps]
                    plt.legend(ps, lps, loc=2)

                    fig.canvas.draw()
                    # Now we can save it to a numpy array.
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    # self._debug_img[self._screen_dimensions[0] // 3 * 2 - 200:self._screen_dimensions[0] // 3 * 2 + 400, self._screen_dimensions[1] // 2 - 800:self._screen_dimensions[1] // 2 + 800] = data
                    self._debug_img = data
                    plt.cla()

                    self._update_screen(self._screen, self._debug_img)

            # wait for next iteration
            rate.sleep()

        # clean shutdown
        self.close()

    @staticmethod
    def close():
        """
        makes it faster to shutdown ros
        """
        rospy.logwarn("close")
        pygame.quit()

    def set_debug_msg(self, msg):
        self._debug_msg = msg

    @staticmethod
    def _get_text_size(text, fontface=cv2.FONT_HERSHEY_COMPLEX, fontscale=2, thickness=2):
        """
        static method to get the text size for a text with given parameters

        :param text:
        :param fontface:
        :param fontscale:
        :param thickness:
        :return:
        """
        return cv2.getTextSize(text, fontface, fontscale, thickness)


if __name__ == '__main__':
    Dashboard()

