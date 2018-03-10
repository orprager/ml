import numpy as np


class SVM:
    # def __init__(self, visualization=True):
    def __init__(self):
        self.w = None
        self.b = None
        self.data_dimension = None
        self.labels_map = {}

    def _fit_check_input(self, train_data, train_label):
        assert type(train_data) == np.ndarray
        assert type(train_label) == np.ndarray
        assert (np.unique(train_label).ravel()).shape[0] == 2  # check that we have only two classes to classify
        # TODO: support more than 2 classes

    def _get_max_feature_values(self, data):
        # Here we find the max feature value in order to initial our w vector as an optimum
        max_feature = None

        for feature_set in data:
            for feature in feature_set:
                if not max_feature or feature > max_feature:
                    max_feature = feature

        return max_feature

    # train
    def fit(self, train_data, train_label, b_range_multiple=5, b_multiple_step=2, optimization_steps=3,
            opt_steps_multiplier=0.1):

        self._fit_check_input(train_data, train_label)
        self.data_dimension = train_data.shape[1]

        # Assuming there are only two labels, set label 1 to 1 and the second one to -1
        unique_labels = np.unique(train_label).ravel()
        train_label[train_label == unique_labels[0]] = 1
        train_label[train_label == unique_labels[1]] = -1

        # { ||w||: [w,b] }
        opt_dict = {}

        max_feature_value = self._get_max_feature_values(train_data)

        # None: PROBLEMATIC WITH HIGH DIMENSION, therefor we create only 4 transforms
        transforms = np.array([np.random.choice([-1, 1], size=(self.data_dimension,)) for i in xrange(4)])

        # support vectors yi(xi.w+b) = 1
        step_sizes = [max_feature_value*opt_steps_multiplier]
        for i in xrange(1, optimization_steps):
            step_sizes.append(step_sizes[i - 1] * opt_steps_multiplier)

        latest_optimum = max_feature_value

        # in order to avoid w with norm bigger that those we already found we initialize min_w_norm var and update
        # it's value by the most smaller w size
        min_w_norm = 999999999

        for step in step_sizes:
            w = np.array([latest_optimum for n in xrange(self.data_dimension)])

            optimized = False
            while not optimized:
                for b in np.arange(-1 * (max_feature_value * b_range_multiple),
                                   max_feature_value * b_range_multiple,
                                   step * b_multiple_step):
                    # The transformation matrix let us check diff w directions
                    for transformation in transforms:
                        w_t = w * transformation
                        w_norm = np.linalg.norm(w_t)
                        if w_norm in opt_dict or w_norm > min_w_norm:
                            continue
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
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
                            if w_norm < min_w_norm:
                                min_w_norm = w_norm
                            opt_dict[w_norm] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            if norms:
                # ||w|| : [w,b]
                opt_choice = opt_dict[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                latest_optimum = opt_choice[0][0] + step * 2
            else:
                print "failed to find proper w and b"
                return

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
