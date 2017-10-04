import glob
import csv
import pandas as pd

src = '/Users/jakobkammerer/Learning/carnd/'

def import_data(root_path='/Users/jakobkammerer/Google Drive/Happy Robots/'):
    filepaths_real = glob.glob(src + 'TrafficLightData_real/*.png')
    filepaths_sim = glob.glob(src + 'TrafficLightData_sim/*.png')

    labels_real = read_labels_from_csv(src + 'TrafficLightData_real/state.csv')
    labels_sim = read_labels_from_csv(src + 'TrafficLightData_sim/state.csv')

    print("REAL: {} Pictures, {} Labels".format(len(filepaths_real), len(labels_real)))
    print("SIM:  {} Pictures, {} Labels".format(len(filepaths_sim), len(labels_sim)))

    #df_sim = to_dataframe(filepaths_sim, labels_sim)
    #print(df_sim.head())
    #return filepaths_real


def read_labels_from_csv(path_to_csv):
    with open(path_to_csv) as file:
        csv_read = csv.reader(file)
        labels = []
        [labels.append(int(line[3])) for line in csv_read]

    return labels

def to_dataframe(file_paths, labels):
    df = pd.DataFrame([file_paths, labels])
    return df

test = import_data()
