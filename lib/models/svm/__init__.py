import matplotlib.pyplot as plt
# from matplotlib import style
import numpy as np
import itertools


# style.use('ggplot')


class SVM:
    # def __init__(self, visualization=True):
    def __init__(self):
        self.w = None
        self.b = None
        self.data_dimension = None
        # self.visualization = visualization
        # self.colors = {1: 'r', -1: 'b'}
        # if self.visualization:
        #     self.fig = plt.figure()
        #     self.ax = self.fig.add_subplot(1, 1, 1)

    def _fit_check_input(self, train_data, train_label):
        assert type(train_data) == np.ndarray
        assert type(train_label) == np.ndarray
        assert (np.unique(train_label).ravel()).shape[0] == 2  # check that we have only two classes to classify
        # TODO: support more than 2 classes

    def _get_min_max_feature_values(self, data):
        min_feature = None
        max_feature = None

        for featureset in data:
            for feature in featureset:
                if not min_feature or feature < min_feature:
                    min_feature = feature

                if not max_feature or feature > max_feature:
                    max_feature = feature

        return max_feature, min_feature


    # train
    def fit(self, train_data, train_label):

        self._fit_check_input(train_data, train_label)

        # { ||w||: [w,b] }
        opt_dict = {}

        self.data_dimension = train_data.shape[1]
        transforms = list(list(t) for t in itertools.product([1, -1], repeat=self.data_dimension))

        max_feature_value, min_feature_value = self._get_min_max_feature_values(train_data)

        # support vectors yi(xi.w+b) = 1

        step_sizes = [max_feature_value * 0.1,
                      max_feature_value * 0.01,
                      # point of expense:
                      max_feature_value * 0.001, ]

        # extremely expensive
        b_range_multiple = 5
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum for n in xrange(self.data_dimension)])

            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (max_feature_value * b_range_multiple),
                                   max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    # The transformation matrix let us check diff w directions
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # 
                        # #### add a break here later..
                        for datapoint, label in zip(train_data, train_label):
                            # yi(w.x - b) >= 1
                            # we're iterating on be from negative values to positive
                            if not label * (np.dot(w_t, datapoint) + b) >= 1:
                                # means that one of the data points is not valid by our constraint.
                                # our constraint is that each point will be in the side it was labeled at,
                                # if it's not happening for our hyperplane it's not a valid hyperplane divider.
                                found_option = False
                                break

                        if found_option:
                            # we found hyperplane that divide our classes perfectly
                            # now, we'll store w and b in order to find the best divider (minimum ||w||)
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def _predict_check_input(self, features):
        assert type(features) == np.ndarray, "Predict expect ndarray of features as input"
        assert self.b is not None
        assert self.w is not None
        assert self.data_dimension is not None
        assert features.shape[1] == self.data_dimension

    def _predict_classification(self, features):
        return np.sign(np.dot(np.array(features), self.w) + self.b)

    def predict(self, predict_data):

        self._predict_check_input(predict_data)

        return np.apply_along_axis(self._predict_classification, 1, predict_data)

