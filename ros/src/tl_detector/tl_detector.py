#!/usr/bin/env python
import yaml
from cv_bridge import CvBridge

import cv2
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from styx_msgs.msg import TrafficLightArray, TrafficLight
from utilities import utils
import numpy as np

from light_classification.tl_classifier import TLClassifier

# TODO Remove? STATE_COUNT_THRESHOLD = 3
MIN_VIS_DIST = 0.0
MAX_VIS_DIST = 100.0

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector', log_level=rospy.WARN)

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_line_waypoints = []
        self.stop_line_waypoints_base_indices = []

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_state_array = np.zeros(5, dtype=np.uint8)
        self.last_wp = -1
        self.state_count = 0

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            # Get index and state of the next traffic light
            index, state = self.process_traffic_lights()

            # TODO For testing: Car should stop at second traffic light
            # if index == 753:
            #     state = self.lights[0].state

            rospy.loginfo("Next traffic light wp index={}, state={}".format(index, state))

            # Publish index of upcoming traffic light if its status is red or yellow
            if state == TrafficLight.GREEN or state == TrafficLight.UNKNOWN:
                index = -1
            self.upcoming_red_light_pub.publish(Int32(index))

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Unsubscribe?
        if self.waypoints is None:
            self.waypoints = waypoints

    def traffic_cb(self, msg):
        # TODO Unsubscribe?
        self.lights = msg.lights

    def image_cb(self, msg):
        self.camera_image = msg

    def get_light_state(self):
        """Determines the current color of the traffic light shown in the camera image

        Returns:
            int: Modal value for the last x classifications

        """
        if self.camera_image is None:
            return TrafficLight.UNKNOWN
        else:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # TODO use light location to zoom in on traffic light in image
            if self.lights is not None:
                print('Simulator Light State {}'.format(self.lights[0].state))

            new_state = np.uint8(self.light_classifier.get_classification(cv_image))

            # Delete first item in last state array
            self.last_state_array = np.delete(self.last_state_array, 0)
            self.last_state_array = np.append(self.last_state_array, new_state)

            modal_value = np.argmax(np.bincount(self.last_state_array))

            rospy.logerr("Predicted: {}; Modalwert: {}; Last State Array {}".format(new_state, modal_value,
                                                                                    self.last_state_array))

            return modal_value

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.pose is not None and self.waypoints is not None:
            if len(self.stop_line_waypoints) == 0:
                for i in range(len(self.stop_line_positions)):
                    # Calculate waypoint indices for stop lines
                    stop_line_pose = PoseStamped()
                    stop_line_pose.pose.position.x = self.stop_line_positions[i][0]
                    stop_line_pose.pose.position.y = self.stop_line_positions[i][1]
                    stop_line_waypoint = Waypoint()
                    stop_line_waypoint.pose = stop_line_pose
                    self.stop_line_waypoints.append(stop_line_waypoint)
                    self.stop_line_waypoints_base_indices.append(utils.get_next(stop_line_pose,
                                                                                self.waypoints.waypoints,
                                                                                skip_orientation_check=True)[0])

                    rospy.logdebug("sl_pose.x={}, y={}".format(stop_line_pose.pose.position.x,
                                                               stop_line_pose.pose.position.y))

            # Get waypoint index of next stopline and its distance from current pose
            index_stop_line, distance = utils.get_next(self.pose, self.stop_line_waypoints)
            # If within visibility range classify image and return waypoint index along with its state
            # If outside visibility range return Unknown as state
            if MIN_VIS_DIST <= distance <= MAX_VIS_DIST:
                return self.stop_line_waypoints_base_indices[index_stop_line], self.get_light_state()
            else:
                return self.stop_line_waypoints_base_indices[index_stop_line], TrafficLight.UNKNOWN

        else:
            if not self.waypoints:
                rospy.logdebug("No base_waypoints available.")
            if not self.pose:
                rospy.logdebug("No self.pose available.")
            return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
