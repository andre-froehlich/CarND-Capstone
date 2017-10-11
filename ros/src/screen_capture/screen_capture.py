#!/usr/bin/env python

from cv_bridge import CvBridge
import cv2
import rospy
from datetime import datetime
from sensor_msgs.msg import Image


class ScreenCapture(object):
    def __init__(self):
        rospy.init_node('screen_capture_node')

        self.path = '../../../capture/'
        self.image_suffix = '.jpg'
        self.camera_image = None

        self.bridge = CvBridge()

        # subscribe to camera image
        rospy.Subscriber('/image_color', Image, self.image_cb)

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.save_image()
            # prevent same image from being saved again
            self.camera_image = None
            rate.sleep()

    def save_image(self):
        if self.camera_image is not None:
            image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            image_path = self.path + str(datetime.now()) + self.image_suffix
            rospy.logwarn('saving image {}'.format(image_path))
            cv2.imwrite(image_path, image)

    def image_cb(self, image):
        self.camera_image = image


if __name__ == '__main__':
    ScreenCapture()
