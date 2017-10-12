import sys
import glob
import pandas as pd
import numpy as np
import sklearn.utils
import cv2
# from keras.utils import to_categorical

from augmentation import *

# src = '/Users/jakobkammerer/Learning/carnd/'


def import_data(root_path, source, fformat):
    # Strings to directories
    states = ['0-red', '1-yellow', '2-green', '3-nolight'] # 4-twilight
    data_frames = []

    print root_path + source + '0-red' + "/" + fformat

    for state in states:
        # Get image path and label
        filepaths = glob.glob(root_path + source + state + "/" + fformat)
        states = [int(state[0]),] * len(filepaths)

        # Create temporary data frame
        d = {'file_path': filepaths, 'state': states}
        df = pd.DataFrame(data=d)

        data_frames.append(df)

    # print("# Data Import")
    # print("# - from {}".format(root_path))
    # print("# - {} Pictures\n".format(len(data)))

    return data_frames


def get_dataset(df):
    """
    Converts pandas data frame to an array with [file_path, label]
    :param df: pandas.DataFrame input with file_paths, labels
    :return: dataset as array
    """
    file_paths = df['file_path'].as_matrix()
    labels = df['state'].as_matrix()

    dataset = [[file_paths[i], labels[i]] for i in range(len(file_paths))]

    return dataset


def balance_dataset(df):
    """
    Balance the dataset to have equal counts of labels
    :param df: pandas dataset with file paths, labels and source
    :return: pandas dataset balanced
    """
    print('# Data Set Balancing')


    # Init balanced data frame
    df_bal = pd.DataFrame

    states = np.unique(df['state'])

    # Look for minimum occurance of a state in set
    print("# - Analyzing")
    min_glob = 99999999999
    min_state = 6
    for state in states:
        min_temp = len(df.loc[df['state'] == state])
        print("# -- State {}: {} counts".format(state, min_temp))

        if min_temp < min_glob:
            min_glob = min_temp
            min_state = state

    if min_glob == 99999999999:
        print("# > Data set already balanced")

    else:
        print("# - Balancing to {} counts due to state {}".format(min_glob, min_state))

        for state in states:
            df_temp = df.loc[df['state'] == state]

            if len(df_temp) > min_glob:
                reduce = len(df_temp) - min_glob
                # Shuffle the data frame
                df_temp = df_temp.sample(frac=1)
                # Delete rows according to reduce
                df_temp = df_temp.iloc[reduce:]

                if df_bal.empty:
                    df_bal = df_temp
                else:
                    df_bal = pd.concat([df_bal, df_temp], ignore_index=True)

                print("# -- State {} reduced by {}; New counts: {}".format(state, reduce, len(df_temp)))

            else:
                if df_bal.empty:
                    df_bal = df_temp
                else:
                    df_bal = pd.concat([df_bal, df_temp], ignore_index=True)

        print("# > New length of balanced data set: {}\n".format(len(df_bal)))

    return df_bal


def generator_v2(samples_by_state, batch_size=32, augment=True, resize=None):
    for samples in samples_by_state:
        sklearn.utils.shuffle(samples)

    while True:
        X = []
        y = []

        for i in range(batch_size):
            sample_class = np.random.randint(4)
            sample_index = np.random.randint(len(samples_by_state[sample_class]))
            path = samples_by_state[sample_class][sample_index]
            image = cv2.imread(path)
            if resize is not None:
                image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)
            if augment:
                image = augmentation_pipeline(image,
                                              brightness=True,
                                              # color=True,
                                              # shadow=True,
                                              # blur=True,
                                              shift=True,
                                              rot=True,
                                              perspective=True,
                                              aug_prob=0.75,
                                              flip_prob=0.5
                                              )
            y_onehot = [0] * 4
            y_onehot[sample_class] = 1

            X.append(image)
            y.append(y_onehot)

        X, y = sklearn.utils.shuffle(X, y)
        yield np.array(X), np.array(y)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Keep generator running
        # Shuffle the set
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            # Create a subset according to batch_size
            batch_samples = samples[offset:(offset+batch_size)]

            images = []
            labels = []

            for batch_sample in batch_samples:
                # Load image and label
                image = cv2.imread(batch_sample[0])
                image = resize_img(image)
                label = batch_sample[1]

                if image is None:
                    sys.exit("image is None")

                # Augment image
                image_aug = augmentation_pipeline(image)

                images.append(image_aug)
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)

            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield X_train, np.resize(y_train, (batch_size, 4))


def resize_img(img):
    return cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)