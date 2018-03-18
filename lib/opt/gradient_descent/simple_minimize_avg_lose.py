import numpy as np
from matplotlib import pyplot as plt


def plot_line(y, data_points):
    x_values = [i for i in range(int(min(data_points)) - 1, int(max(data_points) + 2))]
    y_values = [y(x) for x in x_values]
    plt.plot(x_values, y_values, 'r')


def summation(y, x_points, y_points):
    diff_sum_yi_to_line = 0
    diff_sum_yi_to_line_multi_x_val = 0

    for i in range(1, len(x_points)):
        diff_yi_to_line = y(x_points[i]) - y_points[i]

        # Sum the diff between our tested line (m and b) and the real y[i] point
        diff_sum_yi_to_line += diff_yi_to_line

        # Sum the diff multiplied with the x value,
        diff_sum_yi_to_line_multi_x_val += diff_yi_to_line * x_points[i]

    avg_lose_diff_yi_to_line = diff_sum_yi_to_line / len(x_points)
    avg_lose_diff_yi_to_line_multi_x = diff_sum_yi_to_line_multi_x_val / len(x_points)

    return avg_lose_diff_yi_to_line, avg_lose_diff_yi_to_line_multi_x


if __name__ == '__main__':
    x_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 15, 16]
    y_points = [1, 1, 3, 3, 6, 3, 7, 7, 8, 9, 13, 14]

    m = 0
    b = 0
    y = lambda x: m * x + b

    learn = .0086  # .001, .01, .1, 1 ...
    stop_at_dist = 0.0001

    previous_avg_lose = None
    previous_avg_lose_multiplied_x = None

    for i in range(1000):
        avg_lose, avg_lose_multiplied_x = summation(y, x_points, y_points)

        if not previous_avg_lose:
            previous_avg_lose = avg_lose
        elif i > 0 and abs(avg_lose) > abs(previous_avg_lose):
            break

        if not previous_avg_lose_multiplied_x:
            previous_avg_lose_multiplied_x = avg_lose_multiplied_x
        elif i > 0 and abs(avg_lose_multiplied_x) > abs(previous_avg_lose_multiplied_x):
            break

        previous_m = m
        previous_b = b

        m = m - learn * avg_lose_multiplied_x
        b = b - learn * avg_lose

        if abs(m - previous_m) <= stop_at_dist and abs(b - previous_b) <= stop_at_dist:
            break

    plot_line(y, x_points)
    plt.plot(x_points, y_points, 'bo')

    print "finished"

