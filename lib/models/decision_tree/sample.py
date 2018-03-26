import numpy

from lib.models.decision_tree import DecisionTree

if __name__ == "__main__":
    training_data = numpy.array([['Green', 3, 'Apple'],
                                   ['Yellow', 3, 'Apple'],
                                   ['Red', 1, 'Grape'],
                                   ['Red', 1, 'Grape'],
                                   ['Yellow', 3, 'Lemon']])

    header = ["color", "diameter", "label"]

    tree = DecisionTree()
    tree.fit(training_data, header, 2)
    print tree
    # q = Question(column_num=0, column_name=header[0], value="Green")
