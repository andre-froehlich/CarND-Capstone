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

from light_classification.tl_classifier import TLClassifier

STATE_COUNT_THRESHOLD = 3

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
        self.last_wp = -1
        self.state_count = 0
        # self.has_image = False

        self.collect_training_data = rospy.get_param('~collect_training_data', False)
        self.training_data_counter = 0
        if self.collect_training_data:
            rospy.logwarn("Collecting training data!")
            self.state_file = open("../../../training_data/state.txt", "w")

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.collect_training_data:
                self.save_data()

            index, state = self.process_traffic_lights()

            # TODO For testing: Car should stop at second traffic light
            if index == 753:
                state = TrafficLight.RED
            # End testing

            rospy.loginfo("Next traffic light wp index={}, state={}".format(index, state))
            if state == TrafficLight.GREEN or state == TrafficLight.UNKNOWN:
                index = -1
            self.upcoming_red_light_pub.publish(Int32(index))

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

        # Plot current pose
        # monitor = False
        # if monitor:
        #     rospy.logerr("Seq: {}: X: {}, Y: {}, Orientation: {}".format(
        #         self.pose.header.seq,
        #         self.pose.pose.position.x,
        #         self.pose.pose.position.y,
        #         self.pose.pose.orientation.w))

    def waypoints_cb(self, waypoints):
        # TODO: Unsubscribe?
        if self.waypoints is None:
            self.waypoints = waypoints
        # rospy.logdebug("X: {}".format(self.waypoints.waypoints[0].pose.pose.position.x))

    def traffic_cb(self, msg):
        # Unsubscribe?
        # if self.lights is None:
        #     rospy.logwarn(msg)
        self.lights = msg.lights

    def save_data(self):
        if self.pose is None:
            return
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        index, dist = utils.get_next(self.pose, self.lights)
        traffic_light = self.lights[index]
        # Through out cases where traffic light is only partially visible?
        if 21.0 < dist < 220.0:
            state = traffic_light.state
        else:
            state = 4

        cv2.imwrite("../../../training_data/data{:06d}.png".format(self.training_data_counter), cv_image)
        self.state_file.write("{},{},{},{}\n".format(self.training_data_counter, index, dist, state))
        self.state_file.flush()

        self.training_data_counter += 1

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.camera_image = msg
        # self.has_image = True

        # light_wp, state = self.process_traffic_lights()
        #
        # '''
        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it. Otherwise the previous stable state is
        # used.
        # '''
        # if self.state != state:
        #     self.state_count = 0
        #     self.state = state
        # elif self.state_count >= STATE_COUNT_THRESHOLD:
        #     self.last_state = self.state
        #     light_wp = light_wp if state == TrafficLight.RED else -1
        #     self.last_wp = light_wp
        #     # self.upcoming_red_light_pub.publish(Int32(light_wp))
        # else:
        #     pass
        #     # self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        # self.state_count += 1

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link", "/world", now, rospy.Duration(1.0))
            trans, rot = self.listener.lookupTransform("/base_link", "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        # TODO Use transform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return x, y

    # def get_light_state(self, light):
    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # if not self.has_image:
        #     self.prev_light_loc = None
        #     return False

        if self.camera_image is None:
            return TrafficLight.UNKNOWN
        else:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # x, y = self.project_to_image_plane(light.pose.pose.position)
            # TODO use light location to zoom in on traffic light in image

            # Get classification
            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Init flag for detected light
        # light = None
        # checker = True if self.waypoints else False

        # List of positions that correspond to the line to stop in front of for a given intersection
        # stop_line_positions = self.config['stop_line_positions']
        # rospy.logerr("Fist Stop Light: X: {}, Y: {}".format(stop_line_positions[0][0], stop_line_positions[0][1]))

        if self.pose is not None and self.waypoints is not None:
            if len(self.stop_line_waypoints) == 0:
                for i in range(len(self.stop_line_positions)):
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


            index_stop_line, _ = utils.get_next(self.pose, self.stop_line_waypoints)
            return self.stop_line_waypoints_base_indices[index_stop_line], self.get_light_state()

            # next_tl = self.lights[index_next_tl].pose.pose

            # Find closest stop line to traffic light
            # index_next_stop_line = utils.get_closest_stop_line(next_tl, self.stop_line_positions)
            # next_stop_line = self.stop_line_positions[index_next_stop_line]

            # monitor = False
            # if monitor:
            #     rospy.logerr("")
            #     rospy.logerr("Current Position of Car: X: {}, Y: {}".format(self.pose.pose.position.x,
            #                                                                 self.pose.pose.position.y))
            #     rospy.logerr(
            #         "Closest Waypoint to Car: X: {}, Y: {}".format(car_position.position.x, car_position.position.y))
            #     rospy.logerr("Closest Traffic Light:   X: {}, Y: {}".format(next_tl.position.x, next_tl.position.y))
            #     rospy.logerr("Closest Stop Line:       X: {}, Y: {}".format(next_stop_line[0], next_stop_line[1]))
        else:
            if not self.waypoints:
                rospy.logdebug("No base_waypoints available.")
            if not self.pose:
                rospy.logdebug("No self.pose available.")
            return -1, TrafficLight.UNKNOWN

        # if light:
        #     state = self.get_light_state(light)
        #     return light_wp, state




if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
