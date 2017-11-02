#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import Debug
from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node', log_level=rospy.DEBUG)

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        kp = rospy.get_param('~kp', 1.0)
        ki = rospy.get_param('~ki', 0.0)
        kd = rospy.get_param('~kd', 0.0)
        is_site_launch = rospy.get_param('is_site_launch', True)
        rospy.loginfo("Is launched in site mode: {}".format(is_site_launch))

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)
        self.debug_publisher = rospy.Publisher('/debug_msg', Debug, queue_size=1)

        # Create `TwistController` object
        self.controller = Controller(vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                                     wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, kp, ki, kd)

        # Subscribe to all the topics you need to
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)

        # Members
        self.dbw_enabled = not is_site_launch
        self.current_velocity = None
        self.twist_cmd = None
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steer = 0.0

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            if self.dbw_enabled and self.current_velocity != None and self.twist_cmd != None:
                throttle, brake, steer = self.controller.control(self.twist_cmd, self.current_velocity)
                self.publish(throttle, brake, steer)

            elif not self.dbw_enabled:
                self.controller.reset()

            rate.sleep()

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg

    def current_velocity_cb(self, msg):
        self.current_velocity = msg

    def twist_cmd_cb(self, msg):
        self.twist_cmd = msg

    def publish(self, throttle, brake, steer):
        rospy.loginfo("publish throttle={} brake={} steer={}".format(throttle, brake, steer))

        throttle_command = ThrottleCmd()
        throttle_command.enable = True
        throttle_command.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        throttle_command.pedal_cmd = throttle
        self.throttle_pub.publish(throttle_command)

        steering_command = SteeringCmd()
        steering_command.enable = True
        steering_command.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(steering_command)

        brake_command = BrakeCmd()
        brake_command.enable = True
        brake_command.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        brake_command.pedal_cmd = brake
        self.brake_pub.publish(brake_command)


if __name__ == '__main__':
    DBWNode()
