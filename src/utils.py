from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_project_data_dir() -> Path:
    return get_project_root().joinpath('data')


def get_project_models_dir() -> Path:
    return get_project_root().joinpath('models')


def get_project_results_dir() -> Path:
    return get_project_root().joinpath('results')
