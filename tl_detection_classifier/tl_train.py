from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tl_train_helper import *
from augmentation import *
import cv2

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Load dataset to train and validate on as pandas DataFrame
#root_path = '/Users/jakobkammerer/Google Drive/Happy Robots/train/'
root_path = '/media/student/OS/Users/andre/Google Drive/Happy Robots/train/'
source = 'simulator/'
fformat='*.png'
data_frames = import_data(root_path, source, fformat)
# print(data_frames[0])


# Balance the dataset
# samples_df_bal = balance_dataset(samples_df)

# Get np.arrays with sampels
samples_by_state = []
for i in range(len(data_frames)):
    samples_by_state.append([])
    file_paths = data_frames[i]['file_path'].as_matrix()
    for path in file_paths:
        img = cv2.imread(path)
        samples_by_state[i].append(img)

# print(len(samples_by_state))
# print (len(samples_by_state[0]))
# print type(samples_by_state[0][0])
# print samples_by_state[0][0].shape

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
train_generator = generator_v2(train_samples_by_state, batch_size=batch_size)
val_generator = generator_v2(val_samples_by_state, batch_size=batch_size)

# X, y = next(train_generator)
# print len(X), len(y)
# print X[0].shape, y[0]

X, y = next(train_generator)
for i in range(len(X)):
    state = ""
    for s in y[i]:
        state += str(s)
    cv2.imwrite("generator_output/sample{}_state{}.png".format(i, state), X[i])

exit()

#
#  Model
#
model = Sequential([
    # Normalize
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(600, 800, 3)),

    # Convolutional Layers
    Conv2D(filters=6,
           kernel_size=(5, 5),
           strides=5,
           activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=16,
           kernel_size=(5, 5),
           strides=5,
           activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),

    # Fully connected layers
    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.5),
    Dense(180, activation='relu'),
    Dense(84, activation='relu'),
    Dense(4)])

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# hist = model.fit_generator(train_generator,
#                            samples_per_epoch=len(train_samples) * 2,
#                            validation_data=val_generator,
#                            nb_val_samples=len(val_samples) * 2,
#                            nb_epoch=2)
hist = model.fit_generator(train_generator,
                           steps_per_epoch=1,
                           epochs=10,
                           verbose=1,
                           validation_data=val_generator,
                           validation_steps=1)

# Save the model
model.save('model/model.h5')

print hist.history

# Plot and history
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc='upper right')
plt.show()
plt.savefig('history.png')
