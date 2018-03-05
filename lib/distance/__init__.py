import abc


class abstractstatic(staticmethod):
    __slots__ = ()

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class DistanceCalcMethod(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abstractstatic
    def calc(a, b):
        pass
