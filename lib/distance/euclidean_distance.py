import numpy as np

from lib.distance import DistanceCalcMethod


class EuclideanDistance(DistanceCalcMethod):
    @staticmethod
    def calc(a, b):
        return np.linalg.norm(a - b)
