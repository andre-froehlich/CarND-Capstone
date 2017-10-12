from styx_msgs.msg import TrafficLight
import keras
import keras.models
import h5py
import cv2
import time
import numpy as np


class TLClassifier(object):
    def __init__(self):
        # file has to be in 'tl_detector' folder
        model_path = 'model_00.h5'
        model_file = h5py.File(model_path, mode='r')
        model_version = model_file.attrs.get('keras_version')
        keras_version = str(keras.__version__).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version {}, but the model was built using {}'
                  .format(keras_version, model_version))

        print('loading traffic light detector model')
        self.model = keras.models.load_model(model_path)
        print('loaded :)')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        start = time.time()
        # image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
        #
        # if not self.image_written:
        #     cv2.imwrite("foo.jpg", image)
        #     self.image_written = True

        result = self.model.predict(image[None, :, :, :], batch_size=1)
        prediction = np.argmax(result)
        if prediction == 3:
            prediction = TrafficLight.UNKNOWN
        print('we predicted: {}'.format(prediction))
        print('predict: {} took {}s\n'.format(result, time.time() - start))
        return prediction
