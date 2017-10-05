import sys
import glob
import csv
import pandas as pd
import numpy as np
import sklearn.utils
import cv2

from augmentation import *

#src = '/Users/jakobkammerer/Learning/carnd/'

def import_data(root_path='/Users/jakobkammerer/Learning/carnd/'):
    # Strings to directories
    filepaths_real = glob.glob(root_path + 'TrafficLightData_real/*.png')
    filepaths_sim = glob.glob(root_path + 'TrafficLightData_sim/*.png')

    # Read in CSV data for labeling
    labels_real = read_labels_from_csv(root_path + 'TrafficLightData_real/state.csv')
    labels_sim = read_labels_from_csv(root_path + 'TrafficLightData_sim/state.csv')

    print("Data Loaded from {}".format(root_path))
    print("REAL: {} Pictures, {} Labels".format(len(filepaths_real), len(labels_real)))
    print("SIM:  {} Pictures, {} Labels".format(len(filepaths_sim), len(labels_sim)))

    # Create data frames (pandas)
    df_real = to_dataframe(filepaths_real, labels_real, source='real')
    df_sim = to_dataframe(filepaths_sim, labels_sim, source='simulator')

    df = pd.concat([df_real, df_sim], ignore_index=True)

    #df.to_csv('../test.csv')

    return df


def read_labels_from_csv(path_to_csv):
    with open(path_to_csv) as file:
        csv_read = csv.reader(file)
        labels = []
        [labels.append(int(line[3])) for line in csv_read]

    return labels

def to_dataframe(file_paths, labels, source='unknown'):

    source = [source,] * len(labels)

    d = {'file_path': file_paths, 'state': labels, 'source': source}
    df = pd.DataFrame(data=d)

    return df

def get_dataset(df, source=None):
    """
    Returns a dataset with [file_path, label] according to selected source;
    If no source given: return all
    :param df: pandas.DataFrame input with file_paths, labels, source
    :param source: optional: select source to be included in returned dataset
    :return: dataset as array
    """
    if source:
        df = df.loc[df['source'] == source]

    file_paths = df['file_path'].as_matrix()
    labels = df['state'].as_matrix()

    dataset = [[file_paths[i], labels[i]] for i in range(len(file_paths))]

    return dataset

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
