import sys
import glob
import pandas as pd
import numpy as np
import sklearn.utils
import cv2

from augmentation import *

#src = '/Users/jakobkammerer/Learning/carnd/'


def import_data(root_path='/Users/jakobkammerer/Google Drive/Happy Robots/train/', fformat='.png', source='*/'):

    # Strings to directories
    states = ['0-red', '1-yellow', '2-green', '3-nolight'] # 4-twilight
    data = pd.DataFrame

    for state in states:
        # Get image path and label
        filepaths = glob.glob(root_path + source + state + '/*' + fformat)
        states = [int(state[0]),] * len(filepaths)

        # Create temporary data frame
        d = {'file_path': filepaths, 'state': states}
        df_temp = pd.DataFrame(data=d)

        # Merge data frames
        if data.empty:
            data = df_temp
        else:
            data = pd.concat([data, df_temp], ignore_index=True)


    print("# Data Import")
    print("# - from {}".format(root_path))
    print("# - {} Pictures\n".format(len(data)))

    return data


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
                image_aug = augementation_pipeline(image)

                images.append(image_aug)
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)

            yield sklearn.utils.shuffle(X_train, y_train)


def resize_img(img):
    return cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)