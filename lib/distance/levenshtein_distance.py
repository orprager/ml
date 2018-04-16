
def get_levenshtein_distance(str_a, len_a, str_b, len_b):

    if len_a == 0:
        return len_b

    if len_b == 0:
        return len_a

    if str_a[len_a - 1] == str_b[len_b - 1]:
        # means that the last char on both strings are the same.

        # cost is the amount of required changes to do in order to get one string from another.
        # if the cost is 0 it means that at that position we do not have to change anything.
        cost = 0
    else:
        cost = 1

    return min(get_levenshtein_distance(str_a, len_a - 1, str_b, len_b) + 1,
               get_levenshtein_distance(str_a, len_a, str_b, len_b - 1) + 1,
               get_levenshtein_distance(str_a, len_a - 1, str_b, len_b - 1) + cost)


if __name__ == '__main__':
    assert get_levenshtein_distance("stringA", len("stringA"), "stringB", len("stringB")) == 1
