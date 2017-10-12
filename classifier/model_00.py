from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from keras.regularizers import l2


def get_model(shape_x, shape_y, channels, nb_classes):
    model = Sequential([
        # Normalize
        Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(shape_y, shape_x, channels)),

        # Convolutional Layers
        Conv2D(filters=16,
               kernel_size=5,
               strides=2,
               activation='relu',
               kernel_initializer='he_normal',
               bias_initializer='zeros',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001)),

        MaxPooling2D(pool_size=(2, 2),
                     strides=(1, 1)),

        Conv2D(filters=16,
               kernel_size=5,
               strides=1,
               activation='relu',
               kernel_initializer='he_normal',
               bias_initializer='zeros',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(3, 3),
                     strides=(2, 2)),

        Conv2D(filters=8,
               kernel_size=5,
               strides=1,
               activation='relu',
               kernel_initializer='he_normal',
               bias_initializer='zeros',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2),
                     strides=(1, 1)),

        Conv2D(filters=4,
               kernel_size=5,
               strides=1,
               activation='relu',
               kernel_initializer='he_normal',
               bias_initializer='zeros',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(4, 4),
                     strides=(2, 2)),

        Dropout(0.5),

        # Fully connected layers
        Flatten(),
        Dense(96,
              activation='relu',
              kernel_initializer='he_normal',
              bias_initializer='zeros',
              kernel_regularizer=l2(0.001),
              bias_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(72,
              activation='relu',
              kernel_initializer='he_normal',
              bias_initializer='zeros',
              kernel_regularizer=l2(0.001),
              bias_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(48,
              activation='relu',
              kernel_initializer='he_normal',
              bias_initializer='zeros',
              kernel_regularizer=l2(0.001),
              bias_regularizer=l2(0.001)),
        Dense(nb_classes,
              activation='softmax',
              kernel_initializer='he_normal',
              bias_initializer='zeros')])

    return model