import numpy as np
from matplotlib import pyplot as plt


def plot_line(y, data_points):
    x_values = [i for i in range(int(min(data_points)) - 1, int(max(data_points) + 2))]
    y_values = [y(x) for x in x_values]
    plt.plot(x_values, y_values, 'r')


if __name__ == '__main__':
    x_y_points = np.array(
        [[1, 1], [2, 1], [3, 3], [4, 3], [5, 6], [6, 3], [7, 7], [8, 7], [9, 8], [15, 9], [15, 13], [16, 14]]).T

    x_y_cov_matrix = np.cov(x_y_points[0], x_y_points[1])
    x_var = np.cov(x_y_points[0])

    slope = x_y_cov_matrix[0][1] / x_var

    avg_y = x_y_points[1].ravel().sum() / len(x_y_points[1].ravel())
    avg_x = x_y_points[0].ravel().sum() / len(x_y_points[0].ravel())

    b = avg_y - slope * avg_x

    y = lambda x: slope * x + b

    plot_line(y, x_y_points[0])
    plt.plot(x_y_points[0], x_y_points[1], 'bo')

    print "y = {m} x + ({b})".format(m=slope, b=b)
