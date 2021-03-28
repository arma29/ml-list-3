from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import normalize

from src.utils import get_project_data_dir


def get_raw_data():
    raw_path = get_project_data_dir().joinpath('raw')
    files = [join(raw_path, f)
             for f in listdir(raw_path) if isfile(join(raw_path, f))]
    return files


def get_raw_names():
    raw_path = get_project_data_dir().joinpath('raw')
    files = [f.split('.')[0]
             for f in listdir(raw_path) if isfile(join(raw_path, f))]
    return files


def process_data(dataset):
    data = arff.loadarff(dataset)
    df = pd.DataFrame(data[0])

    # Criando o conjunto de treinamento X_ e valores alvo (classes) y_
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype('str')

    # Normalizando
    X = normalize(X=X, axis=0, norm='max')

    # Criando conjunto de classes
    target_names = np.array(['0', '1'])
    if('cm1' in dataset):
        target_names = np.array(['false', 'true'])

    return X, y, target_names
