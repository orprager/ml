from lib.utils.math import is_numeric


class Question:
    def __init__(self, column_num, column_name, value):
        self.column_name = column_name
        self.column_num = column_num
        self.value = value

    def match(self, row):
        row_val = row[self.column_num]

        if is_numeric(self.value):
            # e.g: the value of the question is 3 and we ask "does the value in the row's column >= 3?"
            return row_val >= self.value
        else:
            # e.g: the value is RED and we ask "does the value in the row's column == RED?"
            return row_val == self.value

    def __repr__(self):
        operator = "=="

        if is_numeric(self.value):
            operator = ">="

        return "Is {column_name} {operator} {value}".format(column_name=self.column_name, operator=operator,
                                                            value=self.value)
