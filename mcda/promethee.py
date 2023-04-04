import typing

import numpy as np

from mcda.common import aggregate_inter_alternative_stats
from mcda.order import get_ordering_from_value_array
from mcda.types import FeatureSpec, Ordering


def calculate_preference(diff: float, spec: FeatureSpec, profile: int) -> float:
    """
    Calculate the preference for a given difference.

    :param diff: The difference.
    :param spec: The feature specification.
    :param profile: The profile.
    :return: The preference.
    """
    if diff > spec.preference_thresholds[profile]:
        return 1
    if diff <= spec.indifference_thresholds[profile]:
        return 0
    return (diff - spec.indifference_thresholds[profile]) / (
        spec.preference_thresholds[profile] - spec.indifference_thresholds[profile]
    )


def calculate_positive_flow(preferences: np.ndarray) -> np.ndarray:
    """
    Calculate the positive flow for a given set of preferences.

    :param preferences: The preferences.
    :return: The positive flow.
    """
    return np.sum(preferences, axis=1)


def calculate_negative_flow(preferences: np.ndarray) -> np.ndarray:
    """
    Calculate the negative flow for a given set of preferences.

    :param preferences: The preferences.
    :return: The negative flow.
    """
    return np.sum(preferences, axis=0)


def get_promethee_one_ordering(
    negative_flow: np.ndarray,
    positive_flow: np.ndarray,
) -> Ordering:
    """
    Get the ordering of the alternatives according to the Promethee I algorithm.

    :param negative_flow: The negative flow.
    :param positive_flow: The positive flow.
    :return: The ordering of the alternatives.
    """
    negative_order = list(np.argsort(negative_flow))
    positive_order = list(np.argsort(positive_flow)[::-1])

    all_elements = set(negative_order + positive_order)

    successor_sets = {
        element: {
            succ_element
            for succ_element in all_elements
            if succ_element in negative_order[negative_order.index(element) + 1 :]
            and succ_element in positive_order[positive_order.index(element) + 1 :]
        }
        for element in all_elements
    }
    common_nodes = []
    added_nodes = set()
    ordered_by_succ_size = sorted(
        successor_sets.items(), key=lambda x: len(x[1]), reverse=True
    )
    for element, successors in ordered_by_succ_size:
        if element in added_nodes:
            continue
        node = [element]
        for another_element, another_successors in successor_sets.items():
            if another_element in added_nodes or another_element == element:
                continue
            if successors == another_successors:
                node.append(another_element)
        added_nodes.update(node)
        common_nodes.append(node)
    return common_nodes


def get_promethee_two_ordering(
    negative_flow: np.ndarray,
    positive_flow: np.ndarray,
) -> Ordering:
    """
    Get the ordering of the alternatives according to the Promethee II algorithm.

    :param net_flow: The net flow.
    :return: The ordering of the alternatives.
    """
    net_flow = positive_flow - negative_flow
    return get_ordering_from_value_array(net_flow, ascending=False)


class PrometheeSolver:
    def __init__(
        self,
        alternatives: np.ndarray,
        feature_specs: typing.List[FeatureSpec],
        order_func: typing.Union[
            typing.Callable[[np.ndarray, np.ndarray], Ordering],
            typing.Literal["promethee_one", "promethee_two"],
        ] = "promethee_one",
    ) -> None:
        """
        :param alternatives: array of alternatives, where each row is an alternative and each column is a criterion.
        :param feature_specs: list of feature specs, one for each criterion.
        :param order_func: function that takes the negative and positive flow and returns the ordering of the alternatives,
            or a string that is either "promethee_one" or "promethee_two" (will use the default implementation), defaults to "promethee_one".
        """
        assert len(alternatives) > 0, "There must be at least one alternative."
        assert len(feature_specs) > 0, "There must be at least one criterion."
        assert (
            len(feature_specs) == alternatives.shape[1]
        ), "The number of criteria must match the number of columns in the alternatives matrix."
        self.alternatives = alternatives
        self.feature_specs = feature_specs
        if order_func == "promethee_one":
            self.order_func = get_promethee_one_ordering
        elif order_func == "promethee_two":
            self.order_func = get_promethee_two_ordering
        else:
            assert callable(order_func)
            self.order_func = order_func
        self.positive_flow = None
        self.negative_flow = None

    def solve(self) -> Ordering:
        """
        Solve the Promethee problem.

        :return: The ordering of the alternatives.
        """
        preferences = aggregate_inter_alternative_stats(
            self.alternatives,
            self.alternatives,
            self.feature_specs,
            stat_callback=calculate_preference,
        )
        self.negative_flow = calculate_negative_flow(preferences)
        self.positive_flow = calculate_positive_flow(preferences)
        return self.order_func(self.negative_flow, self.positive_flow)
