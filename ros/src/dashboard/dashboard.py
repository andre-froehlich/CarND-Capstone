#!/usr/bin/env python

import cv2
import yaml
import numpy as np
from math import sqrt
import pygame
import rospy
from ast import literal_eval as make_tuple
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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

    _current_pose = None
    _base_waypoints = None
    _final_waypoints = None
    _dbw_enabled = False
    _current_velocity = None
    _twist_cmd = None

    # traffic lights per state
    _traffic_lights_per_state = dict()
    _lights = None

    # shows the track in white on black background
    _track_image = None
    # is updated each iteration with new values
    _dashboard_img = None

    _image = None
    _bridge = CvBridge()

    _config = None
    _stop_line_positions = None

    def __init__(self):
        rospy.init_node('dashboard_node')

        self._read_params()

        # calculate screen dimensions from window dimensions
        self._screen_dimensions = (self._window_dimensions[0] * 3, self._window_dimensions[1] * 3)

        # subscribe to pose and waypoints
        rospy.Subscriber('/current_pose', PoseStamped, self._set_current_pose)
        rospy.Subscriber('/base_waypoints', Lane, self._set_base_waypoints)
        rospy.Subscriber('/final_waypoints', Lane, self._set_final_waypoints)

        # subscribe to traffic light and obstacle topics
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self._set_traffic_lights)
        # rospy.Subscriber('/traffic_waypoint', Lane, self._set_traffic_waypoints)
        # rospy.Subscriber('/obstacle_waypoint', Lane, self._set_obstacle_waypoints)

        # is Drive-By-Wire enabled?
        rospy.Subscriber('/dbw_enabled', Bool, self._set_dbw_enabled)

        # subscribe to camera image
        rospy.Subscriber('/image_color', Image, self._set_image)

        # current velocity and twist topic
        # rospy.Subscriber('/current_velocity', TwistStamped, self._set_current_velocity)
        # rospy.Subscriber('/twist_cmd', TwistStamped, self._set_twist_cmd)

        # Load traffic light config
        config_string = rospy.get_param("/traffic_light_config")
        self._config = yaml.load(config_string)

        # List of positions that correspond to the line to stop in front of for a given intersection
        self._stop_line_positions = self._config['stop_line_positions']

        # initialize screen with given parameters from dashboard.launch
        self._init_screen()

        # start loop
        self._loop()

    def _read_params(self):
        # read window dimensions and background color from dashboard.launch
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
        for wp in self._base_waypoints:
            xs.append(wp.pose.pose.position.x)
            #  normalize y values
            ys.append(self._screen_dimensions[1] - (wp.pose.pose.position.y - 1000.))
        vertices = [np.column_stack((xs, ys)).astype(np.int32)]

        # create empty image with screen dimensions
        self._track_image = np.empty((self._screen_dimensions[0], self._screen_dimensions[1], 3), dtype=np.uint8)
        # draw polylines of the track
        cv2.polylines(self._track_image, vertices, True, self._base_waypoints_color, 5)

        # draw initial car position
        self._current_pose = self._base_waypoints[-1].pose
        self._draw_current_position()

    def _draw_current_position(self):
        if self._current_pose is not None:
            x = int(self._current_pose.pose.position.x)
            y = int(self._screen_dimensions[1] - (self._current_pose.pose.position.y - 1000.))
            cv2.circle(self._dashboard_img, (x, y), 10, self._ego_color, -1)
            self._put_position_to_circle((x, y), inside=False, showLine=True)

    def _put_position_to_circle(self, coordinates, inside=True, showLine=False):
        center = (self._screen_dimensions[0] // 2, self._screen_dimensions[0] - 1000.)
        text = "({}/{})".format(coordinates[0], coordinates[1])
        r = (center[0]-coordinates[0], center[1]-coordinates[1])
        mag = sqrt(r[0]**2 + r[1]**2)
        e_r = (r[0]/mag, r[1]/mag)

        dist = 150. if inside else -100.
        if coordinates[0] > 2000.:
            size, _ = self._get_text_size(text)
            dist = size[0] + 50

        ap = (dist * e_r[0], dist * e_r[1])
        p = (int(coordinates[0] + ap[0]), int(coordinates[1] + ap[1]))

        if showLine:
            cv2.line(self._dashboard_img, (int(coordinates[0]), int(coordinates[1])), p, self._base_waypoints_color, 3)

        cv2.putText(self._dashboard_img, text, p, cv2.FONT_HERSHEY_COMPLEX, 2, self._text_shadow_color, 4)
        cv2.putText(self._dashboard_img, text, p, cv2.FONT_HERSHEY_COMPLEX, 2, self._text_color, 2)

    def _draw_traffic_lights(self):
        if self._traffic_lights_per_state is not None:
            for key, val in self._traffic_lights_per_state.iteritems():
                color = TRAFFIC_STATES[key]
                for tl in val:
                    x = int(tl[0])
                    y = int(self._screen_dimensions[1] - (tl[1] - 1000.))
                    cv2.circle(self._dashboard_img, (x, y), 15, color, -1)
                    self._put_position_to_circle((x, y))

    def _draw_dbw_status(self):
        state = RED
        if self._dbw_enabled:
            state = GREEN
        text = "DBW"
        size, baseline = self._get_text_size(text)
        radius = 20
        cv2.putText(self._dashboard_img, text, (self._screen_dimensions[0] - (size[0] + radius + 50), size[1] + 10),
                    cv2.FONT_HERSHEY_COMPLEX, 2, self._text_color, 2)
        cv2.circle(self._dashboard_img, (self._screen_dimensions[0] - (radius // 2 + 50), baseline + (size[1] // 2)),
                   radius, state, -1)

    def _draw_final_waypoints(self):
        if self._final_waypoints is not None and self._dashboard_img is not None:
            xs = list()
            ys = list()
            for wp in self._final_waypoints:
                xs.append(wp.pose.pose.position.x)
                #  normalize y values
                ys.append(self._screen_dimensions[1] - (wp.pose.pose.position.y - 1000.))
            vertices = [np.column_stack((xs, ys)).astype(np.int32)]
            # draw polylines of the track
            cv2.polylines(self._dashboard_img, vertices, False, self._final_waypoints_color, 8)

    def _write_next_traffic_light(self, baseline):
        baseline2 = None
        if self._current_pose is not None and self._base_waypoints is not None and self._lights is not None and self._stop_line_positions is not None:
            # Get car position and its distance to next base waypoint
            index_car_position, distance_to_waypoint = utils.get_next(self._current_pose, self.waypoints)
            car_position = self._base_waypoints[index_car_position].pose.pose

            # rospy.logerr(self.lights)
            index_next_tl, distance_next_tl = utils.get_next(self._current_pose, self._lights)
            next_tl = self._lights[index_next_tl].pose.pose

            dist_tl_text = "Traffic Light #{} comes up in {} m".format(index_next_tl, distance_next_tl)
            size1, baseline1 = self._get_text_size(dist_tl_text)
            cv2.putText(self._dashboard_img, dist_tl_text, (50, baseline + 15 + size1[1]), cv2.FONT_HERSHEY_COMPLEX, 2, self._text_shadow_color, 4)
            cv2.putText(self._dashboard_img, dist_tl_text, (50, baseline + 15 + size1[1]), cv2.FONT_HERSHEY_COMPLEX, 2, self._text_color, 2)

            # Find closest stop line to traffic light
            index_next_stop_line = utils.get_closest_stop_line(next_tl, self._stop_line_positions)
            next_stop_line = self._stop_line_positions[index_next_stop_line]
            distance_next_stop_line = utils.distance2d((car_position.x, car_position.y), next_stop_line)

            dist_hl_text = "Stop Line for Traffic Light #{} in {} m".format(index_next_tl, distance_next_stop_line)
            size2, baseline2 = self._get_text_size(dist_hl_text)
            cv2.putText(self._dashboard_img, dist_hl_text, (50, baseline1 + 15 + size2[1]), cv2.FONT_HERSHEY_COMPLEX, 2,
                        self._text_shadow_color, 4)
            cv2.putText(self._dashboard_img, dist_hl_text, (50, baseline1 + 15 + size2[1]), cv2.FONT_HERSHEY_COMPLEX, 2,
                        self._text_color, 2)

        return baseline2

    def _loop(self):
        # 1Hz should be enough
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self._base_waypoints is not None:

                pygame.event.pump()
                if pygame.key.get_focused():
                    key = pygame.key.get_pressed()

                    if key[pygame.K_ESCAPE]:
                        self.close()

                    if key[pygame.K_0]:
                        self._save_image(0)

                    if key[pygame.K_1]:
                        self._save_image(1)

                    if key[pygame.K_2]:
                        self._save_image(2)

                    if key[pygame.K_4]:
                        self._save_image(4)

                # get copy of track_image
                self._dashboard_img = np.copy(self._track_image)

                self._draw_current_position()

                self._draw_traffic_lights()

                self._draw_dbw_status()

                self._draw_final_waypoints()

                # test text
                header = "Happy Robots"
                size, baseline = self._get_text_size(header)
                cv2.putText(self._dashboard_img, header, (50, 15 + size[1]), cv2.FONT_HERSHEY_COMPLEX, 2, self._text_shadow_color, 4)
                cv2.putText(self._dashboard_img, header, (50, 15 + size[1]), cv2.FONT_HERSHEY_COMPLEX, 2, self._text_color, 2)

                baseline = self._write_next_traffic_light(baseline)
                rospy.logwarn("baseline: {}".format(baseline))

                # update screen with new image and refresh window
                self._update_screen()

            # wait for next iteration
            rate.sleep()

        # clean shutdown
        self.close()

    def _save_image(self, state):
        if self._image is None:
            rospy.logwarn("Cannot save screenshot, because no image was received yet.")
        elif self._lights is None:
            rospy.logwarn("Cannot save screenshot, because no traffic light information was received yet.")
        else:
            time = rospy.Time.now().to_nsec()
            cv_image = self._bridge.imgmsg_to_cv2(self._image, "bgr8")
            cv2.imwrite("../../../training_data/img_time{:21d}_state{:1d}.png".format(time, state), cv_image)
            rospy.logwarn("Saved screenshot with state {} at time {}...".format(state, time))

            index, dist = utils.get_next(self._current_pose, self._lights)
            traffic_light = self._lights[index]

            x, y = utils.project_to_image_plane(traffic_light.pose.pose.position)

            rospy.logwarn("TL index={}".format(index))
            rospy.logwarn("Pixel value for traffic light: x={}, y={}".format(x, y))

    @staticmethod
    def close():
        rospy.logwarn("close")
        pygame.quit()

    def _set_current_pose(self, msg):
        self._current_pose = msg
        rospy.loginfo(
            "Received new position: x={}, y={}".format(self._current_pose.pose.position.x, self._current_pose.pose.position.y))

    def _set_base_waypoints(self, lane):
        if self._base_waypoints is None:
            self._base_waypoints = lane.waypoints
            rospy.logwarn("Waypoints loaded... found {}.".format(len(self._base_waypoints)))
            # draws track image right after setting waypoints
            # this way it only has to be done once
            self._draw_track()

    def _set_final_waypoints(self, lane):
        self._final_waypoints = lane.waypoints
        # rospy.logwarn("Final waypoints received! Got: {}".format(len(self._final_waypoints)))

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

    def _set_image(self, msg):
        self._image = msg

    def _set_traffic_lights(self, msg):
        self._lights = msg.lights
        self._traffic_lights_per_state.clear()
        for tl in msg.lights:
            x_tl = tl.pose.pose.position.x
            y_tl = tl.pose.pose.position.y
            orientation_tl = tl.pose.pose.orientation
            stamp_tl = tl.pose.header.stamp
            state_tl = tl.state
            if state_tl not in self._traffic_lights_per_state:
                self._traffic_lights_per_state[state_tl] = list()
            self._traffic_lights_per_state[state_tl].append((x_tl, y_tl, orientation_tl, stamp_tl))

    @staticmethod
    def _get_text_size(text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, thickness=2):
        return cv2.getTextSize(text, fontFace, fontScale, thickness)


if __name__ == '__main__':
    Dashboard()
