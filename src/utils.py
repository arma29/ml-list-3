from pathlib import Path
import datetime
import time


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_project_data_dir() -> Path:
    return get_project_root().joinpath('data')


def get_project_models_dir() -> Path:
    return get_project_root().joinpath('models')


def get_project_results_dir() -> Path:
    return get_project_root().joinpath('results')


def print_elapsed_time(time_in_seconds):
    delta = datetime.timedelta(seconds=time_in_seconds)
    stringed = "{:.3f}".format(time_in_seconds)
    print(
        f'Tempo do Experimento: {stringed} segundos\n'
        f'segundos - {delta} hh:mm:ss'
    )
