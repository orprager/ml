import numpy as np
from sklearn.metrics import accuracy_score

from lib.models.knn.knn import KNN

if __name__ == '__main__':
    data = np.random.uniform(low=0.0, high=100.0, size=(100, 1))
    labels = np.random.randint(9, size=(100, 1))

    train_data, train_labels, test_data, test_labels = data[:75, :], labels[:75, :], data[75:, :], labels[75:, :]

    knn = KNN()
    knn.fit(train_data, train_labels.ravel())

    predicted_labels = knn.predict(test_data, k_neig=5)

    accuracy = accuracy_score(predicted_labels, test_labels)

    # It's low accuracy because our sample data is randomized

    print "finished, accuracy: {}".format(accuracy)
