"""
This module contains auxiliary functions for the Interferogram class
"""


def sort_list_of_tuples(list_of_tuples, sort_by_idx=0, reverse=False):
    """
    Sorts elements in a list of tuples
    ---
    Parameters
    ---
    list_of_tuples: list
        List of tuples
    sort_by_idx: int, optional
        Number of index to sort by
        E.g. if a tuple consists of two elements and we would like to sort by the second, set to 1
        Default: 0
    reverse: bool, optional
        If True, the sorting is done in ascending order.
        If False - in descending.
        Default is True
    """
    # sort by the parameter_value
    # signal_and_parameter.sort(key=operator.itemgetter(1))
    list_of_tuples.sort(key=lambda x: x[sort_by_idx], reverse=reverse)
    # split it back into sorted
    return zip(*list_of_tuples)

