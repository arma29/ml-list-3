import time
from os.path import isfile

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import get_project_models_dir
from src.classifiers._kmeansbayes import KMeansBayes


def create_dict(data_dict):
    parameters_dict = {
        'measures_lst': [0.3, 0.4, 0.5],
        'magic_number': 5,
        'elapsed_time': 0,
        'X_pos': data_dict['X_pos'],
        'y_pos': data_dict['y_pos'],
        'X_neg': data_dict['X_neg'],
        'y_neg': data_dict['y_neg'],
        'X_pos_n': data_dict['X_pos_n'],
        'X_neg_n': data_dict['X_neg_n'],
        'target_names': data_dict['target_names'],
        'dataset_name': data_dict['dataset_name'],
        'classifier': {
            '0.3': {
                'bests': [12, 2.5]
            },
            '0.4': {
                'bests': [13, 2.1]
            },
            '0.5': {
                'bests': [8, 1.2]
            }
        }
    }

    if("cm1" in parameters_dict['dataset_name']):
        parameters_dict['n_clusters_lst'] = [25, 26, 30]
        parameters_dict['dist_th_lst'] = [1.1, 1.2]
        parameters_dict['classifier']['0.3']['bests'] = [26, 1.1]
        parameters_dict['classifier']['0.4']['bests'] = [25, 1.1]
        parameters_dict['classifier']['0.5']['bests'] = [30, 1.2]

    return parameters_dict


def has_saved_model(dataset_name):
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')

    if(isfile(filename)):
        return True
    else:
        return False


def get_saved_model(dataset_name):
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')
    return joblib.load(filename=filename)


def save_model(parameters_dict, dataset_name):
    filename = get_project_models_dir().joinpath(dataset_name + '.joblib')
    joblib.dump(value=parameters_dict, filename=filename)


def train_model(data_dict):
    dataset_name = data_dict['dataset_name']
    if(has_saved_model(dataset_name)):
        return get_saved_model(dataset_name)

    parameters_dict = create_dict(data_dict)
    target_names = parameters_dict['target_names']

    exp_time = time.time()

    for measure in parameters_dict['measures_lst']:

        X_pos = parameters_dict['X_pos']
        y_pos = parameters_dict['y_pos']
        X_neg = parameters_dict['X_neg']
        y_neg = parameters_dict['y_neg']
        X_pos_n = parameters_dict['X_pos_n']
        X_neg_n = parameters_dict['X_neg_n']

        rd_state = 1

        X_neg_train, X_neg_test, y_neg_train, y_neg_test = \
            train_test_split(X_neg, y_neg, train_size=measure,
                             random_state=rd_state)

        X_test = np.concatenate((X_pos, X_neg_test))
        y_test = np.concatenate((y_pos, y_neg_test))

        X_neg_train_n, X_neg_test_n, y_neg_train, y_neg_test = \
            train_test_split(X_neg_n, y_neg, train_size=measure,
                             random_state=rd_state)

        X_test_n = np.concatenate((X_pos_n, X_neg_test_n))

        parameters_dict['classifier'][str(measure)]['X_test'] = X_test
        parameters_dict['classifier'][str(measure)]['X_test_n'] = X_test_n
        parameters_dict['classifier'][str(measure)]['y_test'] = y_test

        kmeans_bayes = KMeansBayes(
            neg_class=target_names[0],
            pos_class=target_names[1],
            n_clusters=parameters_dict['classifier'][str(measure)]['bests'][0],
            dist_threshold=parameters_dict['classifier'][str(measure)]['bests'][1])

        parameters_dict['classifier'][str(
            measure)]['X_neg_train'] = X_neg_train
        parameters_dict['classifier'][str(
            measure)]['X_neg_train_n'] = X_neg_train_n
        parameters_dict['classifier'][str(
            measure)]['y_neg_train'] = y_neg_train
        parameters_dict['classifier'][str(measure)]['obj'] = kmeans_bayes.fit(
            X_neg_train_n, y_neg_train, X_neg_train)

    parameters_dict['elapsed_time'] = time.time() - exp_time

    save_model(parameters_dict, dataset_name)

    return parameters_dict
