import math

from lib.models.decision_tree.question import Question


def count_unique_values(data, column_num):
    unique_values = {}
    for row in data:
        if row[column_num] not in unique_values:
            unique_values[row[column_num]] = 0
        unique_values[row[column_num]] += 1
    return unique_values


def calc_impurity_gini(classes_cnt):
    number_of_rows = sum(classes_cnt.values())

    success_prob_sum = 0
    for class_name, class_cnt in classes_cnt.items():
        success_prob_sum += ((class_cnt * 1.0 / number_of_rows) ** 2)  # pow 2

    return 1 - success_prob_sum


def get_best_question(questions, all_rows, label_column):
    best_question = ()
    true_group = []
    false_group = []

    all_row_uncertainty = calc_impurity_gini(count_unique_values(all_rows, label_column))
    best_gain = 0

    for q in questions:
        true_rows = []
        false_rows = []
        for row in all_rows:
            if q.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)

        if len(true_rows) == 0 or len(false_rows) == 0:
            continue

        true_rows_class_cnt = count_unique_values(true_rows, label_column)
        false_rows_class_cnt = count_unique_values(false_rows, label_column)

        true_impurity = calc_impurity_gini(true_rows_class_cnt)
        false_impurity = calc_impurity_gini(false_rows_class_cnt)

        # avg_impurity = (true_impurity * len(true_rows) + false_impurity * len(false_rows)) / len(all_rows)

        gain = info_gain(len(true_rows), true_impurity, len(false_rows), false_impurity, all_row_uncertainty)
        if gain >= best_gain:
            best_gain = gain
            best_question = q
            true_group = true_rows
            false_group = false_rows

    return best_gain, best_question, true_group, false_group


def info_gain(left_cnt, left_impurity, right_cnt, right_impurity, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(left_cnt) / (left_cnt + right_cnt)
    return current_uncertainty - p * left_impurity - (1 - p) * right_impurity

class DecisionNode:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, training_data, header, label_column_id):
        self.tree = self.build_tree(training_data, header, label_column_id)

    def build_tree(self, training_data, header, label_column_id):

        """Builds the tree.

            Rules of recursion: 1) Believe that it works. 2) Start by checking
            for the base case (no further information gain). 3) Prepare for
            giant stack traces.
            """

        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        tree = TreeNode(training_data, header, label_column_id)
        tree.partition()
        # gain, question = find_best_split(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if tree.gain == 0:
            return Leaf(training_data, label_column_id)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = tree.true_rows, tree.false_rows

        # Recursively build the true branch.
        true_branch = self.build_tree(true_rows, header, label_column_id)

        # Recursively build the false branch.
        false_branch = self.build_tree(false_rows,  header, label_column_id)

        return DecisionNode(tree.question, true_branch, false_branch)
        #
        # # root node
        # # node = TreeNode(training_data, header, label_column_id)
        # # node.partition()

    def build_node(self):
        pass

    def predict(self):
        pass

class Leaf:
    def __init__(self, rows, label_column_id):
        self.class_count = count_unique_values(rows, label_column_id)

class TreeNode:
    def __init__(self, rows, header, label_column_id):
        self.rows = rows
        self.true_rows = []
        self.true_node = None
        self.false_node = None
        self.false_rows = []
        self.header = header
        self.question = None
        self.gain = 0
        self.label_column_id = label_column_id

    def partition(self):
        values_to_ask_questions_on = set()

        # pull all unique values to ask questions on
        # assuming the labels are in the the data
        for col_id in xrange(0, len(self.rows[0])):
            # for each column

            if col_id != self.label_column_id:
                unique_column_values = count_unique_values(data=self.rows, column_num=col_id).keys()
                [values_to_ask_questions_on.add((col_id, unique_v)) for unique_v in unique_column_values]

        # create a list of questions
        questions = [Question(column_num=col_id, column_name=self.header[col_id], value=v)
                     for (col_id, v) in values_to_ask_questions_on]

        self.gain, self.question, self.true_rows, self.false_rows = get_best_question(questions=questions,
                                                                                      all_rows=self.rows,
                                                                                      label_column=self.label_column_id)
