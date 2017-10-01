#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane
from utilities import utils

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

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.INFO)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Member variables
        self.base_waypoints = None
        self.pose = None
        self.len_waypoints = None
        self.lookahead_wps = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            if self.base_waypoints is not None and self.pose is not None:
                closest_index, _ = utils.get_next(self.pose, self.base_waypoints.waypoints)
                rospy.loginfo("Closed Waypoint index is: {}, x={}, y={}"
                              .format(closest_index, self.base_waypoints.waypoints[closest_index].pose.pose.position.x,
                                      self.base_waypoints.waypoints[closest_index].pose.pose.position.y))

                final_waypoints = None
                if (closest_index < self.len_waypoints - self.lookahead_wps):
                    final_waypoints = self.base_waypoints.waypoints[closest_index:closest_index + self.lookahead_wps]
                else:
                    final_waypoints = self.base_waypoints.waypoints[closest_index:]
                    rest = self.lookahead_wps - (self.len_waypoints - closest_index)
                    final_waypoints += self.base_waypoints.waypoints[:rest]

                rospy.loginfo("Length of final_waypoints is {}".format(len(final_waypoints)))
                assert (len(final_waypoints) == self.lookahead_wps)

                lane = Lane()
                lane.header.stamp = rospy.Time.now()
                lane.header.frame_id = "/world"
                lane.waypoints = final_waypoints
                self.final_waypoints_pub.publish(lane)

                rospy.logdebug("Published final waypoints...")
                rospy.logdebug(lane)

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg
        rospy.loginfo("Received new position: x={}, y={}".format(self.pose.pose.position.x, self.pose.pose.position.y))

    def waypoints_cb(self, waypoints):
        if self.base_waypoints is None:
            self.base_waypoints = waypoints
            self.len_waypoints = len(self.base_waypoints.waypoints)
            self.lookahead_wps = min(LOOKAHEAD_WPS, self.len_waypoints)
            rospy.logwarn("Waypoints loaded... found {}.".format(self.len_waypoints))

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    @staticmethod
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
