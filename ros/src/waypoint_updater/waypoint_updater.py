#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane
from std_msgs.msg import Int32
from utilities import utils
import math
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
NODE_RATE = 5   # 5 Hz
LOOKAHEAD_WPS = 100  # Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS_SITE = 20 # number of waypoints for site; must be lower since there are only ~80 wp total
COMFORTABLE_DECEL = -2.0
MAX_DECEL = -4.0
STOP_CORRIDOR = 6


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self._is_site = rospy.get_param('is_site_launch', False)

        # Member variables
        self.pose = None
        self.base_waypoints = None
        self.working_waypoints = None
        self.len_waypoints = None
        self.wp_dists = []
        self.wp_cum_dist = []
        self.lookahead_wps = None
        self.traffic_waypoint_index = -1
        self.is_braking_active = False
        self.current_velocity = None
        self.last_zeroed_waypoints = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(NODE_RATE)
        while not rospy.is_shutdown():
            if self.working_waypoints is not None and self.pose is not None:
                closest_index, _ = utils.get_next(self.pose, self.working_waypoints.waypoints)
                rospy.logdebug("Closest Waypoint index is: {}, x={}, y={}".format(closest_index,
                        self.working_waypoints.waypoints[closest_index].pose.pose.position.x,
                        self.working_waypoints.waypoints[closest_index].pose.pose.position.y))

                final_waypoints = None
                # Check, how far ahead of current pose is the red traffic light
                rospy.logdebug("tl_index={}, closest_index={}".format(self.traffic_waypoint_index,
                        closest_index))
                is_deepcopy = False
                if self.traffic_waypoint_index != -1:  # red or yellow light ahead
                    if not self.is_braking_active:
                        delta_index = self.traffic_waypoint_index - closest_index
                        if delta_index <= 0:
                            delta_index += self.len_waypoints
                        self.is_braking_active = self.adapt_speed(closest_index, delta_index)
                else:  # green or unknown light ahead
                    if self.is_braking_active:
                        i = closest_index
                        while i != (self.last_zeroed_waypoints + 1):
                            wp = self.base_waypoints.waypoints[i]
                            self.set_waypoint_velocity(self.working_waypoints.waypoints[i], self.get_waypoint_velocity(wp))
                            i += 1
                            if i >= self.len_waypoints:
                                i = 0
                        self.is_braking_active = False

                if closest_index < self.len_waypoints - self.lookahead_wps:
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

    def adapt_speed(self, current_index, delta_index):
        rospy.logdebug("delta_index={}".format(delta_index))

        # Apply stopping corridor and calculate stop_index
        delta_index = max(delta_index - STOP_CORRIDOR, 1)
        stop_index = current_index + delta_index
        if stop_index >= self.len_waypoints:
            stop_index -= self.len_waypoints

        # Determine braking distance
        v0 = self.current_velocity.twist.linear.x
        v0v0 = v0 * v0
        braking_distance = - 0.5 * v0v0 / COMFORTABLE_DECEL

        # Determine waypoint index where braking shall start
        i = stop_index - 1
        if i < 0:
            i = self.len_waypoints - 1
        while True:
            dist = self.wp_cum_dist[stop_index] - self.wp_cum_dist[i]
            if i == current_index or dist >= braking_distance:
                i = current_index + (STOP_CORRIDOR // 2)
                break
            i -= 1
            if i < 0:
                i = self.len_waypoints - 1
        start_index = i

        # Calculate necessary deceleration to stop in time
        a = -0.5 * v0v0 / dist
        rospy.logdebug("Space for braking={}, Deceleration={}".format(dist, a))
        if a < MAX_DECEL:
            rospy.logdebug("Too late to stop. Deceleration exceeds safety limit.")
            return False  # Not braking

        # Set velocity to -0.1 for everything between stop_index and current_index
        # to keep breaking
        # current_index to start_index is left unchanged.
        i = stop_index
        while i != stop_index + self.lookahead_wps:
            self.set_waypoint_velocity(self.working_waypoints.waypoints[i], -0.1)
            i += 1
            if i >= self.len_waypoints:
                i = 0
        self.last_zeroed_waypoints = i

        # Beginning from stop_index with velocity 0
        # iterate backwards and continuously increase velocity until either
        # travel speed or start_index is reached.
        last_speed = 0.0
        i = stop_index - 1
        if i < 0:
            i = self.len_waypoints - 1
        while i != start_index:
            dist = self.wp_dists[i]
            v = math.sqrt(last_speed * last_speed - 2 * dist * a)
            if v < self.get_waypoint_velocity(self.working_waypoints.waypoints[i]):
                self.set_waypoint_velocity(self.working_waypoints.waypoints[i], v)
                last_speed = v
            else:
                break
            i -= 1
            if i < 0:
                i = self.len_waypoints - 1

        return True  # Braking


    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.len_waypoints = len(self.base_waypoints.waypoints)
        self.lookahead_wps = min(LOOKAHEAD_WPS, self.len_waypoints) if not self._is_site else LOOKAHEAD_WPS_SITE
        self.working_waypoints = deepcopy(waypoints)

        cummulated_dist = 0.0
        self.wp_cum_dist = []
        self.wp_dists = []
        for i in range(self.len_waypoints - 1):
            dist = utils.dist(self.base_waypoints.waypoints[i].pose, self.base_waypoints.waypoints[i+1].pose)
            self.wp_dists.append(dist)
            self.wp_cum_dist.append(cummulated_dist)
            cummulated_dist += dist
        last_dist = utils.dist(self.base_waypoints.waypoints[self.len_waypoints-1].pose,
                               self.base_waypoints.waypoints[0].pose)
        self.wp_dists.append(last_dist)
        self.wp_cum_dist.append(cummulated_dist)

        rospy.loginfo("Waypoints loaded... found {}.".format(self.len_waypoints))

    def traffic_cb(self, msg):
        self.traffic_waypoint_index = msg.data

    def current_velocity_cb(self, msg):
        self.current_velocity = msg

    @staticmethod
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
