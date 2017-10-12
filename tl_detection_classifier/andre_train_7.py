from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tl_train_helper import *
from augmentation import *
import cv2
import os

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

import tensorflow

print("Keras version: {}".format(keras.__version__))
print("Tensorflow version: {}".format(tensorflow.__version__))

# Load dataset to train and validate on as pandas DataFrame
# root_path = '/Users/jakobkammerer/Google Drive/Happy Robots/train/'
#root_path = '/media/student/OS/Users/andre/Google Drive/Happy Robots/train/'
root_path = '../training_data/'
#root_path = '../../../train_4/'
source = 'simulator/'
fformat = '*.png'
# source = 'real/'
# fformat = '*.jpg'
data_frames = import_data(root_path, source, fformat)
# print(data_frames[0])
# print(data_frames[1])
# print(data_frames[2])
# print(data_frames[3])

# Balance the dataset
# samples_df_bal = balance_dataset(samples_df)

# Get np.arrays with sampels
samples_by_state = []
for i in range(len(data_frames)):
    samples_by_state.append([])
    file_paths = data_frames[i]['file_path'].as_matrix()
    for path in file_paths:
        # img = cv2.imread(path)
        samples_by_state[i].append(path)

# Split into train and test set
train_samples_by_state = []
val_samples_by_state = []
i = 0
for samples in samples_by_state:
    train_samples, val_samples = train_test_split(samples, test_size=.3)
    train_samples_by_state.append(train_samples)
    val_samples_by_state.append(val_samples)
    print("State: {}".format(i))
    print("Training size={}".format(len(train_samples)))
    print("Validation size={}".format(len(val_samples)))
    print("")
    i += 1

# Set up generators
batch_size = 32
input_x = 800
input_y = 600
train_generator = generator_v2(train_samples_by_state, batch_size=batch_size, resize=(input_x, input_y))
val_generator = generator_v2(val_samples_by_state, batch_size=batch_size, resize=(input_x, input_y))

# X, y = next(train_generator)
# print len(X), len(y)
# print X[0].shape, y[0]

# X, y = next(train_generator)
# for i in range(len(X)):
#     state = ""
#     for s in y[i]:
#         state += str(s)
#     cv2.imwrite("generator_output/sample{}_state{}.png".format(i, state), X[i])
#
# exit()

#
#  Model
#

model = Sequential([
    # Normalize
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(input_y, input_x, 3)),

    # Convolutional Layers
    Conv2D(nb_filter=16,
                  nb_row=5,
                  nb_col=5,
                  subsample=(2, 2),
                  activation='relu',
                  init='he_normal',
                  W_regularizer=l2(0.01),
                  b_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2),
                 strides=(1, 1)),

    Conv2D(nb_filter=16,
           nb_row=5,
           nb_col=5,
            subsample=(1, 1),
           activation='relu',
           init='he_normal',
           W_regularizer=l2(0.01),
           b_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(3, 3),
                 strides=(2, 2)),

    Conv2D(nb_filter=8,
           nb_row=5,
           nb_col=5,
           subsample=(1, 1),
           activation='relu',
           init='he_normal',
           W_regularizer=l2(0.01),
           b_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2),
                 strides=(1, 1)),

    Conv2D(nb_filter=4,
           nb_row=5,
           nb_col=5,
           subsample=(1, 1),
           activation='relu',
           init='he_normal',
           W_regularizer=l2(0.01),
           b_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(4, 4),
                 strides=(2, 2)),

    Dropout(0.5),

    # Fully connected layers
    Flatten(),
    Dense(96,
          activation='relu',
          init='he_normal',
          W_regularizer=l2(0.01),
          b_regularizer=l2(0.01)),
    Dropout(0.5),
    # Dense(256,
    #       activation='relu',
    #       init='he_normal'),
    Dense(72,
          activation='relu',
          init='he_normal',
          W_regularizer=l2(0.01),
          b_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(48,
          activation='relu',
          init='he_normal',
          W_regularizer=l2(0.01),
          b_regularizer=l2(0.01)),
    Dense(4,
          activation='softmax',
          init='he_normal')])

print(model.summary())

# exit()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# hist = model.fit_generator(train_generator,
#                            samples_per_epoch=len(train_samples) * 2,
#                            validation_data=val_generator,
#                            nb_val_samples=len(val_samples) * 2,
#                            nb_epoch=2)

hist = model.fit_generator(train_generator,
                           samples_per_epoch=batch_size * 10,
                           nb_epoch=50,
                           verbose=1,
                           validation_data=val_generator,
                           nb_val_samples=batch_size * 10)

# Save the model
model.save('model/andre_07.h5')

# print hist.history
# Plot and history
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc='upper right')
# plt.show()
plt.savefig('model/history_andre_07.png')

dev0 = os.system("echo -n '\a'")
