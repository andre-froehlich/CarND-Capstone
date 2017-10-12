import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
# from keras.preprocessing.image import ImageDataGenerator
import tensorflow
import cv2
# import numpy as np
import model_00
import generator

print("Keras version: {}".format(keras.__version__))
print("Tensorflow version: {}".format(tensorflow.__version__))

# image_path = '/media/student/OS/Users/andre/Google Drive/Happy Robots/train 3/simulator/'
image_path = '../../../train_4/real/'
filemask = "*.jpg"
# filemask = "*.png"
#states = ['0-red', '1-yellow', '2-green', '3-nolight']
states = ['0-red', '1-yellow', '2-green']

batch_size = 32
model = model_00.get_model(shape_x=800, shape_y=600, channels=3, nb_classes=len(states))
print(model.summary())
model_no = '00_real'

train_data, val_data = generator.get_samples_by_state(image_path, states, filemask)
train_generator = generator.generator(train_data, resize=(800, 600))
val_generator = generator.generator(val_data, resize=(800, 600))

if False:
    example_X, example_y = train_generator.next()
    for i in range(len(example_X)):
        cv2.imwrite("example_images/{}.jpg".format(i), example_X[i])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit_generator(generator=train_generator,
                              steps_per_epoch=10,
                              epochs=50,
                              verbose=1,
                              validation_data=val_generator,
                              validation_steps=10)

model.save('model/model_{}.h5'.format(model_no))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Loss and Accuracy')
plt.ylabel('categorical crossentropy loss/accuracy')
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc='upper right')
plt.savefig('model/model_{}.png'.format(model_no))