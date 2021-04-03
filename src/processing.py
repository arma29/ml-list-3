from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import normalize

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

    # Criando conjunto de classes
    if("datatrieve" in dataset_path):
        df_neg = df.loc[df['Faulty6_1'] == b'0']
        df_pos = df.loc[df['Faulty6_1'] == b'1']
        target_names = np.array(['0', '1'])
    else:  # cm1 is in path
        df_neg = df.loc[df['defects'] == b'false']
        df_pos = df.loc[df['defects'] == b'true']
        target_names = np.array(['false', 'true'])

    # Criando o conjunto  X_ e valores alvo (classes) y_
    X_pos = df_pos.iloc[:, :-1].values
    y_pos = df_pos.iloc[:, -1].values.astype('str')
    X_neg = df_neg.iloc[:, :-1].values
    y_neg = df_neg.iloc[:, -1].values.astype('str')

    # Normalizando
    X_pos_n = normalize(X=X_pos, axis=0, norm='max')
    X_neg_n = normalize(X=X_neg, axis=0, norm='max')

    # Criando dicionário de retorno
    data_dict = {}
    data_dict['X_pos'] = X_pos
    data_dict['y_pos'] = y_pos
    data_dict['X_neg'] = X_neg
    data_dict['y_neg'] = y_neg
    data_dict['X_pos_n'] = X_pos_n
    data_dict['X_neg_n'] = X_neg_n
    data_dict['target_names'] = target_names

    return data_dict
