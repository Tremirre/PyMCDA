import numpy as np

from mcda.types import Ordering


def get_pref_matrix(ordering: Ordering) -> np.ndarray:
    """
    Get the preference matrix from an ordering.

    :param ordering: list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
                        where each preceding node is preferred over the following nodes
    :return: preference matrix
    """
    if len(ordering) == 0:
        return np.array([])
    m_size = max(max(node) for node in ordering) + 1
    pref_matrix = np.zeros((m_size, m_size))
    for i, node in enumerate(ordering):
        for subnode in ordering[i + 1 :]:
            for num in node:
                for next_num in subnode:
                    pref_matrix[num, next_num] = 1
    for node in ordering:
        for num in node:
            for next_num in node:
                if num == next_num:
                    continue
                pref_matrix[num, next_num] = 0.5
    return pref_matrix


def get_ordering_from_value_array(
    value_array: np.ndarray, ascending: bool = False
) -> Ordering:
    """
    Get the ordering of the alternatives according to the array values.
    E.g. if the array is [0.1, 0.2, 0.3, 0.2, 0.1], the ordering will be [[2], [1, 3], [0, 4]] (ascending=False).

    :param value_array: array of values from which the ordering is to be obtained
    :param ascending: whether to sort ascendingly, defaults to False
    :return: ordering
    """
    sorted_arr_args = list(np.argsort(value_array)[::-1])
    sorted_arr = np.sort(value_array)[::-1]
    nodes = []
    for i, (net, arg) in enumerate(zip(sorted_arr, sorted_arr_args)):
        if i == 0 or net != sorted_arr[i - 1]:
            nodes.append([arg])
        else:
            nodes[-1].append(arg)
    if ascending:
        nodes = nodes[::-1]
    return nodes


def get_kendall_distance(ordering_1: Ordering, ordering_2: Ordering) -> float:
    """
    Get the Kendall distance between two orderings.

    :param ordering_1: ordering in the form of a list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
    :param ordering_2: ordering in the form of a list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
    :return: distance between the two orderings
    """
    pref_matrix_1 = get_pref_matrix(ordering_1)
    pref_matrix_2 = get_pref_matrix(ordering_2)
    return np.sum(np.abs(pref_matrix_1 - pref_matrix_2)) / 2


def get_kendall_tau(ordering_1: Ordering, ordering_2: Ordering) -> float:
    """
    Get the Kendall tau between two orderings.

    :param ordering_1: ordering in the form of a list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
    :param ordering_2: ordering in the form of a list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
    :return: kendall tau between the two orderings
    """
    m_size = max(max(node) for node in ordering_1) + 1
    distance = get_kendall_distance(ordering_1, ordering_2)
    return 1 - 4 * (distance / (m_size * (m_size - 1)))
