
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import src.plot_utils as pu
from src.classifiers._kmeansbayes import KMeansBayes
from src.utils import get_project_results_dir


def plot_hq_mtx(parameters_dict):
    measures_lst = parameters_dict['measures_lst']
    dataset_name = parameters_dict['dataset_name']

    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 4.4)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_axisbelow(True)

        curr_name = measures_lst[i]
        title = f'{curr_name*100}\%'

        X_test = parameters_dict['classifier'][str(curr_name)]['X_test']
        y_test = parameters_dict['classifier'][str(curr_name)]['y_test']

        classifier = parameters_dict['classifier'][str(curr_name)]['obj']

        cf_mtx = confusion_matrix(y_test, classifier.predict(X_test))
        tn, fp, fn, tp = cf_mtx.ravel()
        print(f'Title:{title} - TN:{tn} FP:{fp} FN:{fn} TP:{tp}\n')

        disp = ConfusionMatrixDisplay(confusion_matrix=cf_mtx)

        disp.plot(ax=ax, cmap=plt.cm.Blues)

        ax.set_title(title)

    plt.tight_layout()

    filename = get_project_results_dir().joinpath(dataset_name + '_cf_mtx.eps')

    return fig, str(filename)


def plot_massive(parameters_dict):
    dataset_name = parameters_dict['dataset_name']
    target_names = parameters_dict['target_names']
    measures_lst = parameters_dict['measures_lst']

    pu.figure_setup()

    fig_size = pu.get_fig_size(12, 9)
    fig = plt.figure(figsize=(fig_size))
    fig.suptitle(f'Dataset: {dataset_name.upper()}')

    ax = fig.add_subplot()
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.plot([0, 1], [0, 1], 'r--')  # Plot bissectriz

    for measure in measures_lst:
        X_test = parameters_dict['classifier'][str(measure)]['X_test']
        y_test = parameters_dict['classifier'][str(measure)]['y_test']

        X_neg_train = parameters_dict['classifier'][str(
            measure)]['X_neg_train']
        y_neg_train = parameters_dict['classifier'][str(
            measure)]['y_neg_train']

        n_clusters = parameters_dict['classifier'][str(measure)]['bests'][0]
        dist_threshold = parameters_dict['classifier'][str(
            measure)]['bests'][1]

        classifier = KMeansBayes(
            neg_class=target_names[0],
            pos_class=target_names[1],
            n_clusters=n_clusters,
            dist_threshold=dist_threshold
        ).fit(X_neg_train, y_neg_train)

        tn, fp, fn, tp = confusion_matrix(
            y_test, classifier.predict(X_test)).ravel()

        tp_rate = tp/(tp+fn)
        fp_rate = fp/(fp+tn)

        f_score = tp/(tp + (1/2)*(fn+fp))
        f_score = '{:.3f}'.format(f_score)

        auc = (fp_rate*tp_rate + (tp_rate+1)*(1-fp_rate))/2
        auc = '{:.3f}'.format(auc)

        ax.plot(fp_rate, tp_rate,
                label=f'(size = {measure}, k = {n_clusters}, t = {dist_threshold}) AUC = {auc}, F1-measure = {f_score}', marker='o', markersize='3')

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    plt.legend()
    plt.tight_layout()

    filename = get_project_results_dir().joinpath(
        dataset_name + '_mass.eps')

    return fig, str(filename)


def produce_report(parameters_dict):
    fig, filename = plot_hq_mtx(parameters_dict)
    pu.save_fig(fig, filename)
    fig, filename = plot_massive(parameters_dict)
    pu.save_fig(fig, filename)
    plt.show()
