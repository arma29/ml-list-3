from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from sklearn.cluster import KMeans

import math


class KMeansBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, pos_class, neg_class, n_clusters=8, dist_threshold=1.2):
        super().__init__()
        self.__pos_class = pos_class
        self.__neg_class = neg_class
        self.classes_ = [neg_class, pos_class]
        self.n_clusters = n_clusters
        self.dist_threshold = dist_threshold
        self.__is_fitted = False
        self.__dist_dict = {}
        self.__mean_dict = {}
        self.__std_dict = {}
        self.bayes_threshold = 0

    def fit(self, X, y, X_un=None):
        self.__is_fitted = True
        self.__kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=1).fit(X)

        self.__dist_dict = self.__initialize_dist_dict(X)

        if(X_un is None):
            return self

        self.__initialize_stats_dict(X_un)
        self.bayes_threshold = self.__compute_bayes_threshold(X_un)

        return self

    def __initialize_stats_dict(self, X_un):
        # Pré computar medias e stds de cada att
        for i in range(X_un.shape[1]):
            column = X_un[:, i]
            self.__mean_dict[str(i)] = np.mean(column)
            self.__std_dict[str(i)] = np.std(column)

    def __compute_bayes_threshold(self, X_un):

        arr = [self.__compute_bayes_individual(x) for x in X_un]
        return min(arr)

    def __compute_bayes_individual(self, x):
        # Para cada atributo do exemplo
        probs = 1
        for i, att in enumerate(x):
            # Calcular a probabilidade do valor supondo uma gaussiana
            mean = self.__mean_dict[str(i)]
            std = self.__std_dict[str(i)]

            if(std == 0):
                custom_prob = 1
            else:
                # pdf = norm(loc=self.__mean_dict[str(i)],
                #            scale=self.__std_dict[str(i)]).pdf(att)
                fst_term = (1/(math.sqrt(2*math.pi*(std**2))))
                snd_term = math.exp(-(((att - mean)**2)/(2*(std**2))))
                custom_prob = fst_term * snd_term

            probs *= custom_prob

        return probs

    def __initialize_dist_dict(self, X):
        my_dict = {}
        for i, center in enumerate(self.__kmeans.cluster_centers_):
            indexes = [x[0]
                       for x in enumerate(self.__kmeans.labels_) if x[1] == i]
            my_dict[str(i)] = max([np.linalg.norm(x - center)
                                   for x in X[indexes]])
        return my_dict

    def predict(self, X, X_un=None):
        if(not self.__is_fitted):
            raise Exception('Not fitted')

        predict_lst = []

        # Para cada dado de teste (Normalizado e Não-normalizado)
        for x in zip(X, X_un):
            dist_ratio = self.__compute_kmeans_ratio(x[0])
            curr_prob = self.__compute_bayes_individual(x[1])

            if(dist_ratio > self.dist_threshold or curr_prob < self.bayes_threshold):  # Outlier
                predict_lst.append(self.__pos_class)
            else:
                predict_lst.append(self.__neg_class)
            
        return np.array(predict_lst)

    def __compute_kmeans_ratio(self, x):
        # Pegar o centro mais próximo (a label)
        distances = [np.linalg.norm(x - center)
                     for center in self.__kmeans.cluster_centers_]
        label = np.argmin(distances)
        dist_from_center = min(distances)

        max_dist = self.__dist_dict[str(label)]
        if(max_dist == 0):
            max_dist = 1e-8

        return dist_from_center / max_dist

    def score(self, X, y, X_un=None):
        return self.__accuracy_score(y, np.array(self.predict(X, X_un)))

    def __accuracy_score(self, y_true, y_pred):
        if(len(y_true) != len(y_pred)):
            raise Exception('Diff lens')
        hits = [True for (a, b) in zip(y_true, y_pred) if a == b]
        return len(hits)/len(y_true)

    def predict_proba(self, X, X_un=None):
        predict_lst = self.predict(X, X_un)
        return np.array([[1, 0] if x == self.__neg_class else [0, 1] for x in predict_lst])
