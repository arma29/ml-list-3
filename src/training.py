import datetime
import time
from os.path import isfile

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold as KFold

from src.lvq._lvq import LVQ
from src.neighbors._classification import Knn
from src.utils import get_project_models_dir


def create_dict(X, y, target_names, dataset_name):
    parameters_dict = {
        'k_lst': [1, 3],
        'p_lst': range(10, 35, 5),
        'measures_lst': ['LVQ1', 'LVQ2.1', 'LVQ3', 'None'],
        'magic_number': 5,
        'elapsed_time': 0,
        'X': X,
        'y': y,
        'target_names': target_names,
        'dataset_name': dataset_name
    }
    return parameters_dict


def print_elapsed_time(parameters_dict):
    dataset_name = parameters_dict['dataset_name']
    time_in_seconds = parameters_dict['elapsed_time']
    delta = datetime.timedelta(seconds=time_in_seconds)
    print(
        f'{dataset_name} - Tempo do Experimento: {time_in_seconds} \n'
        f'segundos - {delta} hh:mm:ss'
    )


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


def train_model(X, y, target_names, dataset_name):
    if(has_saved_model(dataset_name)):
        return get_saved_model(dataset_name)

    parameters_dict = create_dict(X, y, target_names, dataset_name)
    magic_number = parameters_dict['magic_number']

    exp_time = time.time()

    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    for measure in parameters_dict['measures_lst']:
        # print ('*'*10, dataset_name, '*'*10, '\n')

        processing_time = []
        acc = []
        acc_std = []

        processing_p_time = []
        acc_p = []
        acc_p_std = []

        for k in parameters_dict['k_lst']:

            obj = Knn(n_neighbors=k)

            if(k == parameters_dict['k_lst'][-1] and measure != 'None'):

                for p in parameters_dict['p_lst']:
                    tmp_p_proc_time = []
                    for x in range(magic_number):
                        start_time = time.time()
                        scores = custom_cross_val_score(
                            obj, X, y, cv, measure, p)
                        tmp_p_proc_time.append(time.time() - start_time)

                    processing_p_time.append(np.mean(tmp_p_proc_time))
                    acc_p.append(np.mean(scores))
                    acc_p_std.append(np.std(scores))

                new_key = f'{measure}-p'
                parameters_dict[new_key] = []
                parameters_dict[new_key].extend(
                    [processing_p_time, acc_p, acc_p_std])

            tmp_proc_time = []
            for x in range(magic_number):  # 30 times for statistical relevance
                # Train + Test
                start_time = time.time()
                scores = custom_cross_val_score(
                    obj, X, y, cv, measure, parameters_dict['p_lst'][0])
                tmp_proc_time.append(time.time() - start_time)

            # Saving measurements
            processing_time.append(np.mean(tmp_proc_time))
            acc.append(np.mean(scores))
            # print(f'Acc Cust: {acc[-1]} - k = {k}; measure = {measure}')
            acc_std.append(np.std(scores))

        new_key = f'{measure}-k'
        parameters_dict[new_key] = []
        parameters_dict[new_key].extend([processing_time, acc, acc_std])
        # print(
        #     f'Processing Time: {parameters_dict[measure][0]} \n'
        #     f'- M Number: {magic_number}'
        # )
        # print(f'Acc: {parameters_dict[measure][1]}')
        # print(f'Acc Std: {parameters_dict[measure][2]}')

    parameters_dict['elapsed_time'] = time.time() - exp_time

    save_model(parameters_dict, dataset_name)

    return parameters_dict


def custom_cross_val_score(estimator, X, y, cv, version, p_number):
    scores = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pg = LVQ(prototypes_number=p_number, version=version)
        s_set = pg.generate(X_train, y_train)

        estimator.fit(s_set[0], s_set[1])
        scores.append(estimator.score(X_test, y_test))

    return scores
