import numpy as np

from pymcda.types import Ordering


def calc_srf(ranking: Ordering, relative_importance: float) -> np.ndarray:
    """
    Calculate the SRF for a given ranking and a given min_max_relative_importance.

    :param ranking: The ranking of the features as list of groups.
    :param relative_importance: The min-max relative importance.
    :return: The SRF.
    """
    spaced_weights = np.linspace(1, relative_importance, len(ranking))
    entries_count = max(max(node) for node in ranking if node) + 1
    actual_weights = np.zeros(entries_count)
    for i, node in enumerate(ranking):
        for entry in node:
            actual_weights[entry] = spaced_weights[i]
    normalized_weights = actual_weights / np.sum(actual_weights)
    return normalized_weights
