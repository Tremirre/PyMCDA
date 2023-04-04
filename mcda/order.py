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
