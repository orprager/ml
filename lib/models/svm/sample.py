import numpy as np
import matplotlib.pyplot as plt
import cvxpy
from cvxpy import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lib.models.svm import SVM


def get_oriented_data(test_ratio=0.2, n_dimensions=2):
    # Define the two sets
    m = 10000  # Number of points in each class
    n = 10000

    class1_center = [1, 1]  # E.g. [1,1]
    class2_center = [3, 1]  # E.g. [2,2]

    # Set a seed which will generate feasibly separable sets
    #  Note: these may only be separable with the default tutorial settings
    np.random.seed(8)

    # Define random orientations for the two clusters
    orientation_class1 = np.clip(np.random.randn(n_dimensions, 2), 2, 2)
    orientation_class2 = np.clip(np.random.randn(n_dimensions, 2), 2, 2)

    # Generate unit-normal elements, but clip outliers.
    rclass1 = np.clip(np.random.randn(m, n_dimensions), -2, 2)
    rclass2 = np.clip(np.random.randn(n, n_dimensions), -2, 2)

    class1 = class1_center + np.dot(rclass1, orientation_class1)
    class2 = class2_center + np.dot(rclass2, orientation_class2)

    data = np.concatenate((class1, class2), axis=0)

    labels_c1 = np.full((m,), 1)
    labels_c2 = np.full((n,), -1)
    labels = np.concatenate((labels_c1, labels_c2), axis=0)

    data_train, data_test, labels_train, labels_test = train_test_split(data,
                                                                        labels,
                                                                        test_size=test_ratio,
                                                                        random_state=42,
                                                                        # to preserve initial class balance
                                                                        stratify=labels)

    return data_train, data_test, labels_train, labels_test


if __name__ == '__main__':
    # TODO: currently support only two classes

    train_data, test_data, train_labels, test_labels = get_oriented_data(n_dimensions=10)

    svm = SVM()
    svm.fit(train_data, train_labels.ravel())

    predicted_labels = svm.predict(test_data)
    accuracy = accuracy_score(predicted_labels, test_labels.ravel())

    print "finished, accuracy: {}".format(accuracy)
