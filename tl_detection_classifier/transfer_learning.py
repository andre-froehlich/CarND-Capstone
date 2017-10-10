import os
import sys
import glob
import argparse
import matplotlib

# so we can save images from matplotlib on AWS
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from tl_train_helper import *
from augmentation import *
import pickle

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
INCEPTION_SIZE = (IM_WIDTH, IM_HEIGHT)
NB_EPOCHS = 2
BATCH_SIZE = 32
FC_SIZE = 1024
NB_CLASSES = 4
NB_IV3_LAYERS_TO_FREEZE = 172

root_path = '../training_data/'
source = 'simulator/'
fformat = '*.png'

data_frames = import_data(root_path, source, fformat)

samples_by_state = []
for i in range(len(data_frames)):
    samples_by_state.append([])
    file_paths = data_frames[i]['file_path'].as_matrix()
    for path in file_paths:
        samples_by_state[i].append(path)

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
train_generator = generator_v2(train_samples_by_state, batch_size=BATCH_SIZE, resize=INCEPTION_SIZE, augment=False)
validation_generator = generator_v2(val_samples_by_state, batch_size=BATCH_SIZE, resize=INCEPTION_SIZE, augment=False)


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new fully connected layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def plot_training(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc='upper right')
    plt.savefig('history.png')


def train():
    """Use transfer learning and fine-tuning to train a network on a new dataset"""

    # setup model
    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, NB_CLASSES)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=NB_EPOCHS,
        samples_per_epoch=BATCH_SIZE * 100,
        validation_data=validation_generator,
        nb_val_samples=BATCH_SIZE * 10,
        class_weight='auto')

    # fine-tuning
    setup_to_finetune(model)

    history_ft = model.fit_generator(
        train_generator,
        samples_per_epoch=BATCH_SIZE * 100,
        nb_epoch=NB_EPOCHS,
        validation_data=validation_generator,
        nb_val_samples=BATCH_SIZE * 10,
        class_weight='auto')

    model.save('model.h5')

    # dump history for later use
    pickle.dump(history_ft.history, open("history.pickle", "wb"))

    plot_training(history_ft)


if __name__ == "__main__":
    train()
