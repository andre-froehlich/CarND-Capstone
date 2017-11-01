import os
import csv
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import rospy
import time
from dbw_mkz_msgs.msg import BrakeCmd

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        self._wheel_radius = kwargs['wheel_radius']
        self._brake_deadband = kwargs['brake_deadband']
        self._accel_limit = kwargs['accel_limit']
        self._decel_limit = kwargs['decel_limit']
	self._max_throttle = kwargs['max_throttle']

        vehicle_mass = kwargs['vehicle_mass']
        fuel_capacity = kwargs['fuel_capacity']
        self._total_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self._deadband_torque = BrakeCmd.TORQUE_MAX * self._brake_deadband

        # PID controller used for velocity control
        # using kp, ki, kd from params file
        kp = kwargs['kp']
        ki = kwargs['ki']
        kd = kwargs['kd']
        self._pid = PID(kp, ki, kd, self._decel_limit, self._accel_limit)
        self._last_t = None

        # yaw_controller used for steering angle
        wheel_base = kwargs['wheel_base']
        steer_ratio = kwargs['steer_ratio']
        min_speed = kwargs['min_speed']
        max_lat_accel = kwargs['max_lat_accel']
        max_steer_angle = kwargs['max_steer_angle']
        yaw_args = [wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle]
        self._yaw_controller = YawController(*yaw_args)

        tau = kwargs['tau']
        ts = kwargs['ts']
        self._low_pass_filter = LowPassFilter(tau, ts)

        self._twist_values = [[], [], [], [], [], [], [], [], [], []]
        self._max_data = 200

        rospy.logdebug("wheel_base: {}\tsteer_ratio: {}\tmax_lat_accel: {}\tmax_steer_angle: {}\n".format(wheel_base, steer_ratio, max_lat_accel, max_steer_angle))
        self._log_data = []


    def control(self, twist_cmd, velocity_cmd):
        if self._last_t is None:
            self._last_t = time.time()
            return 0.0, 0.0, 0.0

        # Calculate steering
        twist_linear_x = twist_cmd.twist.linear.x
        twist_angular_z = twist_cmd.twist.angular.z
        current_velocity_x = velocity_cmd.twist.linear.x

        delta_t = self._last_t - time.time()
        self._last_t = time.time()
        # calculating error and acceleration
        error_v = twist_linear_x - current_velocity_x
        a = self._low_pass_filter.filt(self._pid.step(error_v, delta_t))

        # keeping speed but minor error_v
        if 0.0 < error_v < 0.5:
            brake = 0.0
            throttle = min(self._max_throttle, a) * 0.75 if a > 0.0 else 0.0
        elif error_v <= 0 or twist_linear_x < 0:
            brake = -a * self._total_mass * self._wheel_radius

            # keep braking as long as twist velocity < 0
            if brake > 0.0 and twist_linear_x < 0:
                brake = BrakeCmd.TORQUE_MAX * 0.5
            elif brake < self._deadband_torque:
                brake = 0.0
            throttle = 0.0

            rospy.logdebug("twist_linear_x={}\terror_v={}\tbrake={}".format(twist_linear_x, error_v, brake))

        else:
            brake = 0.0
            throttle = min(self._max_throttle, a)

        self._twist_values[0].append(twist_linear_x)
        self._twist_values[1].append(current_velocity_x)
        self._twist_values[2].append(brake)
        self._twist_values[3].append(throttle)
        self._twist_values[4].append(error_v)
        self._twist_values[5].append(self._low_pass_filter.get())

        steer = self._yaw_controller.get_steering(twist_linear_x, twist_angular_z, current_velocity_x)

        self._log_data.append({
            'twist_linear_x': twist_linear_x,
            'current_velocity_x': current_velocity_x,
            'error_v': error_v,
            'throttle': throttle,
            'brake': brake,
            'steer': steer,
            'twist_angular_z': twist_angular_z
        })

        return throttle, brake, steer

    def reset(self):
        self._pid.reset()

    def save(self):
        basepath = os.path.dirname(os.path.abspath(__file__))
        logfile = os.path.join(basepath, 'twist_controller.csv')
        fieldnames = ['twist_linear_x', 'current_velocity_x', 'error_v', 'throttle', 'brake', 'steer', 'twist_angular_z']
        with open(logfile, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._log_data)

