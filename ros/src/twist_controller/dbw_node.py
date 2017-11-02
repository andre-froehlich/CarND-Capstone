#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import Debug
from twist_controller import Controller


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node', log_level=rospy.DEBUG)

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', 0.1)
        decel_limit = rospy.get_param('~decel_limit', -5.0)
        accel_limit = rospy.get_param('~accel_limit', 1.0)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.0)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.0)
        min_speed = rospy.get_param('~min_speed', 1.0)
        max_throttle = rospy.get_param('~max_throttle', 0.025)

        kp = rospy.get_param('~kp', 1.0)
        ki = rospy.get_param('~ki', 0.0)
        kd = rospy.get_param('~kd', 0.0)

        tau = rospy.get_param('~tau', 2.0)
        ts = rospy.get_param('~ts', 1.0)

        is_site_launch = rospy.get_param('is_site_launch', True)
        rospy.loginfo("Is launched in site mode: {}".format(is_site_launch))

        self._steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self._throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self._brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)
        self._debug_publisher = rospy.Publisher('/debug_msg', Debug, queue_size=1)

        params = {
            'vehicle_mass': vehicle_mass,
            'fuel_capacity': fuel_capacity,
            'brake_deadband': brake_deadband,
            'decel_limit': decel_limit,
            'accel_limit': accel_limit,
            'wheel_radius': wheel_radius,
            'wheel_base': wheel_base,
            'steer_ratio': steer_ratio,
            'max_lat_accel': max_lat_accel,
            'max_steer_angle': max_steer_angle,
            'min_speed': min_speed,
            'max_throttle': max_throttle,
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'tau': tau,
            'ts': ts
        }
        # Create `TwistController` object
        self._controller = Controller(**params)

        # Subscribe to all the topics you need to
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self._dbw_enabled_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self._current_velocity_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self._twist_cmd_cb)
        rospy.Subscriber('/tf_init_done', Bool, self._tf_init_done_cb)

        # Members
        self._dbw_enabled = False
        self._tf_init_done = False
        self._current_velocity = None
        self._twist_cmd = None
        self._last_throttle = 0.0
        self._last_brake = 0.0
        self._last_steer = 0.0

        self._loop()

    def _loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            throttle, brake, steer = 0.0, 0.0, 0.0

            if self._dbw_enabled and self._tf_init_done and self._current_velocity != None and self._twist_cmd != None:
                throttle, brake, steer = self._controller.control(self._twist_cmd, self._current_velocity)
            elif not self._dbw_enabled:
                # reset pid controller
                self._controller.reset()
            elif not self._tf_init_done:
                # brake until tensorflow is initialized
                throttle, brake, steer = 0.0, BrakeCmd.TORQUE_BOO * 2.0, 0.0

            self._publish(throttle, brake, steer)
            rate.sleep()

        self._controller.save()

    def _dbw_enabled_cb(self, msg):
        self._dbw_enabled = msg.data

    def _tf_init_done_cb(self, msg):
        rospy.logdebug("tf init done")
        self._tf_init_done = msg

    def _current_velocity_cb(self, msg):
        self._current_velocity = msg

    def _twist_cmd_cb(self, msg):
        self._twist_cmd = msg

    def _publish(self, throttle, brake, steer):
        self._last_throttle = throttle
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self._throttle_pub.publish(tcmd)
        rospy.loginfo("Issued throttle command, value={}".format(throttle))

        self._last_steer = steer
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self._steer_pub.publish(scmd)
        rospy.loginfo("Issued steer command, value={}".format(steer))

        self._last_brake = brake
        bcmd = BrakeCmd()
        bcmd.enable = True
        # braking only works with torque...
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self._brake_pub.publish(bcmd)
        rospy.loginfo("Issued brake command, value={}".format(brake))


if __name__ == '__main__':
    DBWNode()
