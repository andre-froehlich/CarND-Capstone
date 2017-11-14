from math import atan


class YawController(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel=float('inf'), max_steer_angle=8.2):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel

        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle

    def get_angle(self, radius):
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_v, angular_v, current_v):
        steering_wheel_angle = 0.0
        if abs(current_v) > 0.5:
            steering_wheel_angle = self.steer_ratio * atan(self.wheel_base * angular_v / current_v)
        elif abs(linear_v) > 0.1:
            steering_wheel_angle = self.steer_ratio * atan(self.wheel_base * angular_v / linear_v)

        steering_wheel_angle = min(max(steering_wheel_angle, self.min_angle), self.max_angle)

        return steering_wheel_angle