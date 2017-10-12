import glob
import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from augmentation import augmentation_pipeline

def get_samples_by_state(path, states, filemask):
    data_frames = []

    print path + states[0] + "/" + filemask

    for state in states:
        # Get image path and label
        filepaths = glob.glob(path + state + "/" + filemask)
        states = [int(state[0]), ] * len(filepaths)

        # Create temporary data frame
        d = {'file_path': filepaths, 'state': states}
        df = pd.DataFrame(data=d)
        data_frames.append(df)

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
        train_samples, val_samples = train_test_split(samples, test_size=.2)
        train_samples_by_state.append(train_samples)
        val_samples_by_state.append(val_samples)

    return train_samples_by_state, val_samples_by_state


def generator(samples_by_state, batch_size=32, augment=True, resize=None):
    number_of_states = len(samples_by_state)

    for samples in samples_by_state:
        sklearn.utils.shuffle(samples)

    while True:
        X = []
        y = []

        for i in range(batch_size):
            sample_class = np.random.randint(number_of_states)
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
            y_onehot = [0] * number_of_states
            y_onehot[sample_class] = 1

            X.append(image)
            y.append(y_onehot)

        X, y = sklearn.utils.shuffle(X, y)
        yield np.array(X), np.array(y)
