import keras
import h5py
import cv2
import glob
import numpy as np
import ntpath

model_path = '../ros/src/tl_detector/model_00_real.h5'
input_pattern = './832628/*.jpg'
output_path = './832628_out/'

image_paths = glob.glob(input_pattern)

model_file = h5py.File(model_path, mode='r')
model_version = model_file.attrs.get('keras_version')
keras_version = str(keras.__version__).encode('utf8')
if model_version != keras_version:
    print("You are using Keras version {}, but the model was built using {}".format(keras_version, model_version))
model = keras.models.load_model(model_path)

for image_path in image_paths:
    filename = ntpath.basename(image_path)
    print(image_path)
    image = cv2.imread(image_path)

    if image.shape != (800, 600):
        image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)

    result = model.predict(image[None, :, :, :], batch_size=1)

    print(filename)
    print(result)
    prediction = np.argmax(result)
    print(prediction)

    # red, yellow, green - colors are in BGR format
    colors = [(0, 0, 255),
              (0, 255, 255),
              (0, 255, 0)]
    cv2.circle(image, (15, 15), 10, colors[prediction], thickness=-1)
    np.set_printoptions(precision=3, suppress=True)
    cv2.putText(image, str(result[0]), (5, 42), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), lineType=cv2.CV_AA)
    cv2.imwrite(output_path + filename, image)
