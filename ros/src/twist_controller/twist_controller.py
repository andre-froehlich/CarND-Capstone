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
        rospy.logwarn("wheel_base: {}\tsteer_ratio: {}\tmax_lat_accel: {}\tmax_steer_angle: {}\n".format(wheel_base, steer_ratio, max_lat_accel, max_steer_angle))

        self.wheel_radius = wheel_radius
        self.brake_deadband = brake_deadband
        self.deadband_torque = BrakeCmd.TORQUE_MAX * self.brake_deadband
        self.total_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self.decel_limit = decel_limit

        # PID controller used for velocity control
        # using kp, ki, kd from params file
        self.pid = PID(kp, ki, kd, decel_limit, accel_limit)
        self.last_t = None

        # yaw_controller used for steering angle
        self.yaw_controller = YawController(wheel_base, steer_ratio, 1.0, max_lat_accel, max_steer_angle)

        self.low_pass_filter = LowPassFilter(4.0, 1.0)

        self.twist_values = [[], [], [], [], [], [], [], [], [], []]
        self._max_data = 200


    def control(self, twist_cmd, velocity_cmd):
        if self.last_t is None:
            self.last_t = time.time()
            return 0.0, 0.0, 0.0

        # Calculate steering
        twist_linear_x = twist_cmd.twist.linear.x
        twist_angular_z = twist_cmd.twist.angular.z
        current_velocity_x = velocity_cmd.twist.linear.x

        delta_t = self.last_t - time.time()
        self.last_t = time.time()
        # calculating error and acceleration
        error_v = twist_linear_x - current_velocity_x
        a = self.low_pass_filter.filt(self.pid.step(error_v, delta_t))

        # if a < 0:
        if 0.0 < error_v < 0.5:
            # only correct when abs(error_v) greater than 0.5
            brake = 0.0
            throttle = a * 0.75 if a > 0.0 else 0.0
        elif error_v <= 0 or twist_linear_x < 0:
            brake = -a * self.total_mass * self.wheel_radius

            # keep braking as long as twist velocity < 0
            if brake > 0.0 and twist_linear_x < 0:
                brake = BrakeCmd.TORQUE_MAX * 0.5
            elif brake < self.deadband_torque:
                brake = 0.0
            throttle = 0.0

            rospy.logdebug("twist_linear_x={}\terror_v={}\tbrake={}".format(twist_linear_x, error_v, brake))

        else:
            brake = 0.0
            throttle = min(1.0, a)

        self.twist_values[0].append(twist_linear_x)
        self.twist_values[1].append(current_velocity_x)
        self.twist_values[2].append(brake)
        self.twist_values[3].append(throttle)
        self.twist_values[4].append(error_v)
        self.twist_values[5].append(self.low_pass_filter.get())

        steer = self.yaw_controller.get_steering(twist_linear_x, twist_angular_z, current_velocity_x)

        return throttle, brake, steer

    def reset(self):
        self.pid.reset()