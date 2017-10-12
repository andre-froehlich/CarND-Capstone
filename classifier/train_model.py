import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
# from keras.preprocessing.image import ImageDataGenerator
import tensorflow
# import cv2
# import numpy as np
import model_00
import generator

print("Keras version: {}".format(keras.__version__))
print("Tensorflow version: {}".format(tensorflow.__version__))

# image_path = '/media/student/OS/Users/andre/Google Drive/Happy Robots/train 3/simulator'
image_path = '../../../train_4/simulator'
filemask = "*.jpg"
#states = ['0-red', '1-yellow', '2-green', '3-nolight']
states = ['0-red', '1-yellow', '2-green']

batch_size = 8
model = model_00.get_model(shape_x=800, shape_y=600, channels=3)
model_no = '00'

train_data, val_data = generator.get_samples_by_state(image_path, states, filemask)
train_generator = generator.generator(train_data)
val_generator = generator.generator(val_data)




'''
datagen = ImageDataGenerator(width_shift_range=.1,
                             height_shift_range=.1,
                             shear_range=0.05,
                             zoom_range=.1,
                             fill_mode='nearest',
                             horizontal_flip=True,
                             data_format='channels_last')


image_data_gen = datagen.flow_from_directory(image_path,
                                             target_size=(600, 800),
                                             classes=states,
                                             batch_size=batch_size,
                                             color_mode='rgb')

if False:
    X, y = image_data_gen.next()
    y_sum = np.sum(y, axis=0)
    print(y_sum)
    # img_rgb = cv2.cvtColor(X[0], cv2.COLOR_RGB2BGR)

    for i in range(len(X)):
        img = X[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("example_images/{}.jpg".format(i), img)
'''




hist = model.fit_generator(generator=train_generator,
                              steps_per_epoch=2,
                              epochs=3,
                              verbose=2,
                              validation_data=val_generator,
                              validation_steps=2)

model.save('model/model_{}.h5'.format(model_no))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc='upper right')
plt.savefig('model/model_{}.png'.format(model_no))