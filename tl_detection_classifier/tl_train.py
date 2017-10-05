from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tl_train_helper import *
from augmentation import *

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Load dataset to train and validate on
samples_df = import_data()
samples = get_dataset(samples_df, source='sim')
#print(len(samples))

# Split into train and test set
train_samples, val_samples = train_test_split(samples, test_size=.3)
#print(len(train_samples))
#print(len(val_samples))

# Set up generators
train_generator = generator(train_samples, batch_size=32)
val_generator = generator(val_samples, batch_size=32)

#
#  Model
#

model = Sequential()

# Normalize
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(600, 800, 3)))

# Layers and pooling
model.add(Conv2D(filters=6, kernel_size=5, strides=5, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(180, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(5))

model.compile(loss='mse', optimizer='adam')

hist = model.fit_generator(train_generator,
                           samples_per_epoch=len(train_samples) * 2,
                           validation_data=val_generator,
                           nb_val_samples=len(val_samples) * 2,
                           nb_epoch=2)

# Save the model
model.save('../model.h5')

# Plot and history
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('history.png')