import math
from pid import PID
from lowpass import LowPassFilter

class VelocityController(object):
    self._total_mass = None
    self._wheel_radius = None
    self._brake_deadband = None
    self._accel_limit = None
    self._decel_limit = None
    self._accel_filter = None

    def __init(self, *args, **kwargs):
        self._total_mass = kwargs['total_mass']
        self._wheel_radius = kwargs['wheel_radius']
        self._brake_deadband = kwargs['brake_deadband']
        self._accel_limit = kwargs['accel_limit']
        self._decel_limit = kwargs['decel_limit']
        self._accel_torque_limit = self._total_mass * self._accel_limit * self._wheel_radius

        self._accel_filter = LowPassFilter(2.0, 1.0)

    def get_control(self, target_speed, current_speed, delta_t):
        error_v = target_speed - current_speed
        a = error_v / delta_t
        if a > 0.0:
            a = min(self._accel_limit, a) 
        else:
            a = max(self._decel_limit, a)
        a = self._accel_filter.filt(a)

        target_speed = current_speed + a * delta_t
        T = self._total_mass * a * self._wheel_radius
        throttle = min(T / self._accel_torque_limit, 1.0)

        if throttle <= 0.0:
            if abs(throttle) > abs(self._brake_deadband):
                brake = max(abs(torque) * self._accel_torque_limit, 0.0)
                throttle = 0.0
            else: 
                brake = 0.0
                throttle = 0.0

        return throttle, brake
