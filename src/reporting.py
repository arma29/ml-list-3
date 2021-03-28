import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

import src.plot_utils as pu
from src.lvq._lvq import LVQ
from src.neighbors._classification import Knn
from src.utils import get_project_results_dir


def plot_hq_summary(parameters_dict):
    k_lst = parameters_dict['k_lst']
    measures_lst = parameters_dict['measures_lst']
    dataset_name = parameters_dict['dataset_name']

    fmt = ['ro--', 'g^--', 'bs--', 'k*--']

    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 6)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    ax = fig.add_subplot(1, 2, 1)

    ax.set_xlabel('Parâmetro K')
    ax.set_ylabel('Tempo de Processamento (s)')

    ax.set_axisbelow(True)

    for i in range(len(fmt)):
        curr_measure = f'{measures_lst[i]}-k'
        curr_name = measures_lst[i]
        ax.plot(
            k_lst,
            parameters_dict[curr_measure][0],
            fmt[i],
            markersize=1.5,
            linewidth=0.5,
            label=curr_name)
        ax.set_xticks(k_lst)

    plt.legend()
    plt.tight_layout()

    ax = fig.add_subplot(1, 2, 2)

    ax.set_xlabel('Parâmetro K')
    ax.set_ylabel('Acurácia')

    ax.set_axisbelow(True)

    for i in range(len(fmt)):
        curr_measure = f'{measures_lst[i]}-k'
        curr_name = measures_lst[i]
        ax.plot(
            k_lst,
            parameters_dict[curr_measure][1],
            fmt[i],
            markersize=1.5,
            linewidth=0.5,
            label=curr_name)
        ax.set_xticks(k_lst)

    plt.legend()
    plt.tight_layout()

    filename = get_project_results_dir().joinpath(dataset_name + '_summary.eps')

    return fig, str(filename)


def plot_hq_summary_p(parameters_dict):
    p_lst = parameters_dict['p_lst']
    measures_lst = parameters_dict['measures_lst']
    dataset_name = parameters_dict['dataset_name']

    fmt = ['ro--', 'g^--', 'bs--']

    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 6)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    ax = fig.add_subplot(1, 2, 1)

    ax.set_xlabel('Protótipos')
    ax.set_ylabel('Tempo de Processamento (s)')

    ax.set_axisbelow(True)

    for i in range(len(fmt)):
        curr_measure = f'{measures_lst[i]}-p'
        curr_name = measures_lst[i]
        ax.plot(
            p_lst,
            parameters_dict[curr_measure][0],
            fmt[i],
            markersize=1.5,
            linewidth=0.5,
            label=curr_name)
        ax.set_xticks(p_lst)

    plt.legend()
    plt.tight_layout()

    ax = fig.add_subplot(1, 2, 2)

    ax.set_xlabel('Protótipos')
    ax.set_ylabel('Acurácia')

    ax.set_axisbelow(True)

    for i in range(len(fmt)):
        curr_measure = f'{measures_lst[i]}-p'
        curr_name = measures_lst[i]
        ax.plot(
            p_lst,
            parameters_dict[curr_measure][1],
            fmt[i],
            markersize=1.5,
            linewidth=0.5,
            label=curr_name)
        ax.set_xticks(p_lst)

    plt.legend()
    plt.tight_layout()

    filename = get_project_results_dir().joinpath(dataset_name + '_summary_p.eps')

    return fig, str(filename)


def plot_hq_mtx(parameters_dict):
    X = parameters_dict['X']
    y = parameters_dict['y']

    k_lst = parameters_dict['k_lst']
    p_lst = parameters_dict['p_lst']
    measures_lst = parameters_dict['measures_lst']
    dataset_name = parameters_dict['dataset_name']
    target_names = parameters_dict['target_names']

    pu.figure_setup()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, shuffle=True, stratify=y)

    fig_size = pu.get_fig_size(15, 4.4)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_axisbelow(True)

        curr_name = measures_lst[i]

        pg = LVQ(prototypes_number=p_lst[0], version=curr_name)
        s_set = pg.generate(X_train, y_train)

        classifier = Knn(n_neighbors=k_lst[-1]).fit(s_set[0], s_set[1])
        plot_confusion_matrix(classifier, X_test, y_test,
                              display_labels=target_names,
                              ax=ax,
                              cmap=plt.cm.Blues,
                              normalize=None
                              )

        ax.set_title(curr_name)

    plt.tight_layout()

    filename = get_project_results_dir().joinpath(dataset_name + '_cf_mtx.eps')

    return fig, str(filename)


def produce_report(parameters_dict):
    fig, filename = plot_hq_summary(parameters_dict)
    pu.save_fig(fig, filename)
    fig, filename = plot_hq_summary_p(parameters_dict)
    pu.save_fig(fig, filename)
    fig, filename = plot_hq_mtx(parameters_dict)
    pu.save_fig(fig, filename)
    plt.show()
