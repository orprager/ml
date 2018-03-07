import numpy as np
from collections import Counter
import pickle

from lib.distance.euclidean_distance import EuclideanDistance


class KNN(object):
    VALID_DISTANCE_METHODS = ["euclidean"]
    VALID_TIE_SOLVING_METHODS = ["decr_k"]

    INVALID_SOLVE_TIE_METHOD_ERR_MSG = "solve tie method must be one of the following: {}"
    INVALID_K_NEIGHBORS_ERR_MSG = "you must choose k greater than 0"

    def __init__(self):
        self.training_data = None
        self.training_labels = None
        self.fit_executed = False

    def fit(self, training_data, training_labels):

        # TODO: supporting dataframe will allow me to only change the model while moving from sklrean knn and that impl
        assert type(training_data) == np.ndarray
        assert type(training_labels) == np.ndarray

        self.training_data = training_data
        self.training_labels = training_labels
        self.fit_executed = True

    def _predict_check_input(self, solve_tie_method, k_nigh):

        if not self.fit_executed:
            raise Exception("you must run fit with training data and labels before predict")

        assert solve_tie_method in KNN.VALID_TIE_SOLVING_METHODS, \
            KNN.INVALID_SOLVE_TIE_METHOD_ERR_MSG.format(','.join(KNN.VALID_TIE_SOLVING_METHODS))

        assert k_nigh > 0, KNN.INVALID_K_NEIGHBORS_ERR_MSG

    def predict(self, test_data, k_neig, distance_method=EuclideanDistance, solve_tie_method='decr_k'):

        self._predict_check_input(solve_tie_method, k_neig)

        predicted_labels = []
        for v in test_data:
            neighbors = []
            for t_data, t_label in zip(self.training_data, self.training_labels):

                dist = distance_method.calc(v, t_data)

                # Note: sort efficiency in python -
                # https://stackoverflow.com/questions/8021314/can-i-build-a-list-and-sort-it-at-the-same-time
                neighbors.append((dist, t_label))

            # We sort the neighbors outside of _get_most_common_label because it's recursive method and we don't want so
            # sort more than once.
            sorted_neighbors = sorted(neighbors, key=lambda tup: tup[0])
            predicted_labels.append(self._get_most_common_label(sorted_neighbors, k_neig))

        return predicted_labels

    def _get_most_common_label(self, neighbors, k):
        k_neighbors = neighbors[:k]

        label_counter = Counter(elem[1] for elem in k_neighbors)
        sorted_labels_by_cnt = label_counter.most_common()

        if len(sorted_labels_by_cnt) == 1:
            most_common_label = sorted_labels_by_cnt.pop()[0]
        else:
            if sorted_labels_by_cnt[0][1] > sorted_labels_by_cnt[1][1]:
                most_common_label = sorted_labels_by_cnt[0][0]
            else:
                most_common_label = self._get_most_common_label(neighbors, k-1)

        return most_common_label
