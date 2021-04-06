from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

from src.utils import get_project_data_dir


def get_data_paths():
    raw_path = get_project_data_dir().joinpath('raw')
    files = [join(raw_path, f)
             for f in listdir(raw_path) if isfile(join(raw_path, f))]
    return files


def get_data_names():
    raw_path = get_project_data_dir().joinpath('raw')
    files = [f.split('.')[0]
             for f in listdir(raw_path) if isfile(join(raw_path, f))]
    return files


def process_data(dataset_path):
    data = arff.loadarff(dataset_path)
    df = pd.DataFrame(data[0])

    # Criando o conjunto de treinamento X_ e valores alvo (classes) y_
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype('str')

    # Normalizando
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # Criando conjunto de classes
    if("datatrieve" in dataset_path):
        target_names = np.array(['0', '1'])
    else:  # cm1 is in path
        target_names = np.array(['false', 'true'])

    for j, t in enumerate(target_names):
        indexes = [i for i, yi in enumerate(y) if yi == t]
        if(j == 0):
            X_neg = X[indexes]
            y_neg = y[indexes]
        else:
            X_pos = X[indexes]
            y_pos = y[indexes]

    # Criando dicionário de retorno
    data_dict = {}
    data_dict['X_pos'] = X_pos
    data_dict['y_pos'] = y_pos
    data_dict['X_neg'] = X_neg
    data_dict['y_neg'] = y_neg
    data_dict['target_names'] = target_names

    return data_dict
