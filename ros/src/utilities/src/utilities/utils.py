#!/usr/bin/env python

import rospy
import tf
import math

# Constants
PI_0_5 = math.pi * 0.5
PI_1_0 = math.pi
PI_1_5 = math.pi * 1.5
PI_2_0 = math.pi * 2.0


def get_next(base_pose, pose_list, skip_orientation_check=False):
    """
    Returns index of the next list entry to base_pose
    :param base_pose: Single Pose (e.g. current pose)
    :param pose_list: List with poses to search for the closest
    :return: Index of closest list entry and distance
    """
    closest_dist = float("inf")
    closest_index = 0

    for i in range(0, len(pose_list)):
        # Check if pose in list in in front of the vehicle
        if skip_orientation_check or check_is_ahead(base_pose.pose, pose_list[i].pose.pose):

            # Calculate the distance between pose und pose in list
            dist = squared_dist(base_pose, pose_list[i].pose)

            # If distance is smaller than last saved distance
            if dist < closest_dist:
                # Save
                closest_dist = dist
                closest_index = i
    return closest_index, math.sqrt(closest_dist)


# TODO Used by dashboard, but could we reuse get_next?
def get_closest_stop_line(tl_pose, tl_list):
    """
    Finds the closest stop line to the traffic light
    :param tl_pose: pose of traffic light
    :param tl_list: pose of stop line according to tl_pose
    :return: index of list entry
    """
    closest_dist = float("inf")
    closest_index = 0

    for i in range(0, len(tl_list)):
        # Check if ahead (probably not necessary

        # Calculate the distance between tl_pose and poses in list
        dx = tl_pose.position.x - tl_list[i][0]
        dy = tl_pose.position.y - tl_list[i][1]
        dist = dx * dx + dy * dy

        if dist < closest_dist:
            # Save
            closest_dist = dist
            closest_index = i

    return closest_index


TOLERANCE = 0.001


def is_close(a, b):
    return abs(a - b) < TOLERANCE


# TODO This is not working
def check_is_ahead_2(pose_1, pose_2):
    # Transformation from quaternion to euler
    quaternion = (pose_1.orientation.x,
                  pose_1.orientation.y,
                  pose_1.orientation.z,
                  pose_1.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]

    if is_close(yaw, PI_0_5):
        return pose_2.position.y >= pose_1.position.y
    elif is_close(yaw, PI_1_5):
        return pose_2.position.y < pose_1.position.y
    else:
        m = math.tan(euler[2])
        m_ = -1.0 / m
        b_ = pose_1.position.y - m_ * pose_1.position.x
        left_facing = (PI_0_5 < yaw < PI_1_5)
        p2y_ = m_ * pose_2.position.x + b_

        if (left_facing):
            if m >= 0:
                return p2y_ <= pose_2.position.y
            else:
                return p2y_ > pose_2.position.y
        else:
            if m >= 0:
                return p2y_ >= pose_2.position.y
            else:
                return p2y_ < pose_2.position.y


def check_is_ahead(pose_1, pose_2):
    """
    Checks if pose_2 is in front of the vehicle (pose_1)
    :param pose_1: must (directly) contain position and orientation
    :param pose_2: must (directly) contain position and orientation
    :return: True / False
    """
    # Distances in x and y
    dx = pose_2.position.x - pose_1.position.x
    dy = pose_2.position.y - pose_1.position.y

    # Init angle
    angle = None

    # Quadrant definition
    if dx == 0:
        if dy >= 0:
            angle = 0.5 * PI_1_0
        else:
            angle = PI_1_5
    elif dx > 0.0 and dy >= 0.0:
        angle = math.atan(dy / dx)
    elif dx > 0.0 >= dy:
        angle = PI_2_0 - math.atan(-dy / dx)
    elif dx < 0.0 and dy <= 0.0:
        angle = PI_1_0 + math.atan(dy / dx)
    else:
        angle = PI_1_0 - math.atan(-dy / dx)

    # Transformation from quaternion to euler
    quaternion = (pose_1.orientation.x,
                  pose_1.orientation.y,
                  pose_1.orientation.z,
                  pose_1.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)

    car_angle = euler[2]
    # Normalize orientation
    while car_angle < 0:
        car_angle += PI_2_0
    while car_angle > PI_2_0:
        car_angle -= PI_2_0

    assert (not 0 > car_angle and car_angle <= PI_2_0)

    delta_angle = abs(angle - car_angle)
    if delta_angle >= PI_0_5:
        if delta_angle <= PI_1_5:
            return False
        else:
            return True
    else:
        return True


def dist(pose_1, pose_2):
    return math.sqrt(squared_dist(pose_1, pose_2))


def squared_dist(pose_1, pose_2):
    dx = pose_1.pose.position.x - pose_2.pose.position.x
    dy = pose_1.pose.position.y - pose_2.pose.position.y
    return dx * dx + dy * dy


# TODO Used by dashboard, do we really need this?
def distance2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
