from styx_msgs.msg import TrafficLight
import keras
import keras.models
import h5py
import cv2
import time
import numpy as np
import rospy


class TLClassifier(object):
    def __init__(self, modelpath='model_00.h5'):
        # configure path in launch file, either tl_detector_site.launch or tl_detector.launch
        model_path = modelpath
        model_file = h5py.File(model_path, mode='r')
        model_version = model_file.attrs.get('keras_version')
        keras_version = str(keras.__version__).encode('utf8')

        if model_version != keras_version:
            rospy.logwarn("You are using Keras version {}, but the model was built using {}"
                          .format(keras_version, model_version))

        self.model = keras.models.load_model(model_path)
        rospy.loginfo("Loaded traffic light detector model.")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        start = rospy.Time.now().to_nsec()

        if image.shape != (800, 600):
            image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)

        result = self.model.predict(image[None, :, :, :], batch_size=1)
        prediction = np.argmax(result)

        if prediction == 3:
            prediction = TrafficLight.UNKNOWN

        rospy.loginfo("Traffic Light prediction result={} prediction={} time={}ms"
                      .format(result, prediction, int(rospy.Time.now().to_nsec() - start) / 1000000))

        return prediction
