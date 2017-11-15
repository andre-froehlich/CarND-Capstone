from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import rospy
import time
from dbw_mkz_msgs.msg import BrakeCmd

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius,
                 wheel_base, steer_ratio, max_lat_accel, max_steer_angle, kp, ki, kd):
        rospy.logdebug(
            "wheel_base: {}\tsteer_ratio: {}\tmax_lat_accel: {}\tmax_steer_angle: {}\n".format(wheel_base, steer_ratio,
                                                                                               max_lat_accel,
                                                                                               max_steer_angle))

        self.wheel_radius = wheel_radius
        self._steer_ratio = steer_ratio
        self._max_steer_angle = max_steer_angle
        self.brake_deadband = brake_deadband
        self.deadband_torque = BrakeCmd.TORQUE_MAX * self.brake_deadband
        self.total_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.last_t = None
        self.twist_values = [[], [], [], [], [], [], [], [], [], []]

        # PID controller used for velocity control
        self.pid = PID(kp, ki, kd, decel_limit, accel_limit)

        # yaw_controller used for steering angle
        self.yaw_controller = YawController(wheel_base, steer_ratio, max_lat_accel, max_steer_angle)

        self.low_pass_filter = LowPassFilter(3.5, 1.0)

    def control(self, twist_cmd, current_velocity, actual):
        if self.last_t is None:
            self.last_t = time.time()
            return 0.0, 0.0, 0.0

        # calculate time
        delta_t = self.last_t - time.time()
        self.last_t = time.time()

        # Calculate steering
        twist_linear_x = twist_cmd.twist.linear.x
        twist_angular_z = twist_cmd.twist.angular.z
        current_velocity_x = current_velocity.twist.linear.x
        steer = self.yaw_controller.get_steering(twist_linear_x, twist_angular_z, actual.linear.x)
        steer += 2.0 * (twist_angular_z - actual.angular.z)

        # calculating error and acceleration
        error_v = twist_linear_x - current_velocity_x
        a = self.low_pass_filter.filt(self.pid.step(error_v, delta_t))

        if 0.0 < error_v < 0.5:
            brake = 0.0
            throttle = self.accel_limit // 2
            rospy.logdebug("controller: throttle={}\tbrake={}\tsteer={} (keeping speed)".format(throttle, brake, steer))

        elif error_v <= 0 or twist_linear_x < 0 or a < 0.0:
            brake = -a * self.total_mass * self.wheel_radius

            # keep braking as long as twist velocity < 0
            if abs(brake) > 0.0 and twist_linear_x < 0:
                brake = BrakeCmd.TORQUE_MAX * 0.5
            elif abs(brake) < self.deadband_torque:
                # brake using engine only
                brake = 0.0
            throttle = 0.0

            rospy.logdebug("controller: throttle={}\tbrake={}\tsteer={} (braking)".format(throttle, brake, steer))

        else:
            brake = 0.0
            throttle = min(self.accel_limit, a)
            rospy.logdebug("controller: throttle={}\tbrake={}\tsteer={} (accelerate)".format(throttle, brake, steer))

        self.twist_values[0].append(twist_linear_x)
        self.twist_values[1].append(current_velocity_x)
        self.twist_values[2].append(brake)
        self.twist_values[3].append(throttle)
        self.twist_values[4].append(error_v)
        self.twist_values[5].append(self.low_pass_filter.get())

        return throttle, brake, steer

    def reset(self):
        self.pid.reset()
