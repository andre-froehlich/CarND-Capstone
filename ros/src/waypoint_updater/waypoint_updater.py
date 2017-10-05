#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane
from std_msgs.msg import Int32
from utilities import utils
import math
import sys
from copy import deepcopy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
BRAKING_DISTANCE = 50
MAX_DECEL = -5.0
STOP_CORRIDOR = 10

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Member variables
        self.pose = None
        self.base_waypoints = None
        self.working_waypoints = None
        self.len_waypoints = None
        self.lookahead_wps = None
        self.traffic_waypoint_index = None
        self.is_braking_active = False
        self.current_velocity = 0.0

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            if self.working_waypoints is not None and self.pose is not None:
                closest_index, _ = utils.get_next(self.pose, self.working_waypoints.waypoints)
                rospy.loginfo("Closed Waypoint index is: {}, x={}, y={}"
                              .format(closest_index, self.working_waypoints.waypoints[closest_index].pose.pose.position.x,
                                      self.working_waypoints.waypoints[closest_index].pose.pose.position.y))

                final_waypoints = None
                # Check, how far ahead of current pose is the red traffic light
                rospy.logwarn("tl_index={}, closest_index={}".format(self.traffic_waypoint_index, closest_index))
                delta_index = sys.maxint
                if self.traffic_waypoint_index is not None:
                    delta_index = self.traffic_waypoint_index - closest_index
                    if (delta_index <= 0):
                        delta_index += self.len_waypoints

                if self.is_braking_active:
                    if delta_index > BRAKING_DISTANCE:
                        self.working_waypoints = deepcopy(self.base_waypoints)
                        self.is_braking_active = False
                else:
                    if delta_index <= BRAKING_DISTANCE:
                        self.adapt_speed(closest_index, delta_index)
                        self.is_braking_active = True

                if (closest_index < self.len_waypoints - self.lookahead_wps):
                    final_waypoints = self.working_waypoints.waypoints[closest_index:closest_index + self.lookahead_wps]
                else:
                    final_waypoints = self.working_waypoints.waypoints[closest_index:]
                    rest = self.lookahead_wps - (self.len_waypoints - closest_index)
                    final_waypoints += self.working_waypoints.waypoints[:rest]


                rospy.loginfo("Length of final_waypoints is {}".format(len(final_waypoints)))
                assert (len(final_waypoints) == self.lookahead_wps)

                lane = Lane()
                lane.header.stamp = rospy.Time.now()
                lane.header.frame_id = "/world"
                lane.waypoints = final_waypoints
                self.final_waypoints_pub.publish(lane)

                rospy.loginfo("Published final waypoints...")
                i = 0
                for wp in lane.waypoints:
                    rospy.logdebug("Index={}, velocity={}".format(i, wp.twist.twist.linear.x))
                    i += 1

            rate.sleep()

    def adapt_speed(self, start_index, delta_index):
        rospy.logwarn("delta_index={}".format(delta_index))

        # Apply stopping corridor and calculate stop_index
        delta_index = max(delta_index - STOP_CORRIDOR, 1)
        stop_index = start_index + delta_index
        if stop_index >= self.len_waypoints:
            stop_index -= self.len_waypoints

        # Calculate distances to stop index
        next_distances = [None] * self.len_waypoints
        cummulated_dist = 0.0
        i_last = stop_index
        i = stop_index - 1
        if i < 0:
            i = self.len_waypoints - 1
        while True:
            dist = utils.dist(self.working_waypoints.waypoints[i].pose, self.working_waypoints.waypoints[i_last].pose)
            cummulated_dist += dist
            next_distances[i] = dist
            i_last = i
            i -= 1
            if (i < 0):
                i = self.len_waypoints - 1
            if (i == start_index):
                break

        # Calculate neccessary decelleration to stop in time
        v0 = self.current_velocity.twist.linear.x
        v1 = 0.0
        s1 = cummulated_dist
        a = (v0 * v1 - v0 * v0 + 0.5 * (v1 - v0) ** 2) / s1

        rospy.loginfo("v0={}, v1={}, s1={}, a={}".format(v0, v1, s1, a))

        if a < MAX_DECEL:
            rospy.logwarn("Too late to stop. Needed decelleration would be {}.".format(a))
            return

        # Set velocity to 0.0 for everything between stop_index and start_index
        i = stop_index
        while i != start_index:
            self.set_waypoint_velocity(self.working_waypoints.waypoints[i], 0.0)
            i += 1
            if (i >= self.len_waypoints):
                i = 0

        # Continuously decrease velocity so it reaches 0.0 at delta_index
        last_speed = 0.0
        i = stop_index - 1
        if i < 0:
            i = self.len_waypoints - 1
        while i != start_index:
            dist = next_distances[i]
            v = math.sqrt(last_speed * last_speed - 2 * dist * a)

            if (v < self.get_waypoint_velocity(self.working_waypoints.waypoints[i])):
                self.set_waypoint_velocity(self.working_waypoints.waypoints[i], v)
                last_speed = v
            else:
                break

            i -= 1
            if (i < 0):
                i = self.len_waypoints - 1


    def pose_cb(self, msg):
        self.pose = msg
        # rospy.loginfo("Received new position: x={}, y={}".format(self.pose.pose.position.x, self.pose.pose.position.y))

    def waypoints_cb(self, waypoints):
        if self.base_waypoints is None:
            self.base_waypoints = waypoints
            self.len_waypoints = len(self.base_waypoints.waypoints)
            self.lookahead_wps = min(LOOKAHEAD_WPS, self.len_waypoints)
            self.working_waypoints = deepcopy(self.base_waypoints)
            rospy.logwarn("Waypoints loaded... found {}.".format(self.len_waypoints))
            # rospy.logdebug(waypoints)

    def traffic_cb(self, msg):
        self.traffic_waypoint_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def current_velocity_cb(self, msg):
        self.current_velocity = msg

    @staticmethod
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

    # @staticmethod
    # def set_waypoint_velocity2(waypoints, waypoint, velocity):
    #     waypoints[waypoint].twist.twist.linear.x = velocity


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
