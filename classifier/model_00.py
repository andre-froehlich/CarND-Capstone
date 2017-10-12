from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from keras.regularizers import l2


def get_model(shape_x, shape_y, channels):
    model = Sequential([
        # Normalize
        Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(shape_y, shape_x, channels)),

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

    return model