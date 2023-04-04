import typing

import numpy as np

from dataclasses import dataclass, field
from functools import partial

from typing import List


@dataclass
class FeatureSpec:
    name: str
    is_cost_type: bool
    indifference_thresholds: List[float]
    preference_thresholds: List[float]
    veto_thresholds: List[float] = field(default_factory=lambda: [np.inf])
    weight: float = 1.0


def calc_srf(ranking: List[List[int]], relative_importance: float) -> np.ndarray:
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


def calculate_concordance(diff: float, spec: FeatureSpec, profile: int) -> float:
    """
    Calculate the concordance for a given difference.

    :param diff: The difference.
    :param spec: The feature specification.
    :param profile: The profile.
    :return: The concordance.
    """
    if diff >= -spec.indifference_thresholds[profile]:
        return 1
    if diff < -spec.preference_thresholds[profile]:
        return 0
    return (spec.preference_thresholds[profile] + diff) / (
        spec.preference_thresholds[profile] - spec.indifference_thresholds[profile]
    )


def calculate_discordance(diff: float, spec: FeatureSpec, profile: int) -> float:
    """
    Calculate the discordance for a given difference.

    :param diff: The difference.
    :param spec: The feature specification.
    :param profile: The profile.
    :return: The discordance.
    """
    if diff <= -spec.veto_thresholds[profile]:
        return 1
    if diff >= -spec.preference_thresholds[profile]:
        return 0
    return (-diff - spec.preference_thresholds[profile]) / (
        spec.veto_thresholds[profile] - spec.preference_thresholds[profile]
    )


def aggregate_inter_alternative_stats(
    first_alternatives: np.ndarray,
    second_alternatives: np.ndarray,
    feature_specs: List[FeatureSpec],
    stat_callback: typing.Callable[[float, FeatureSpec, int], float],
    average_out: bool = True,
    move_profiles: bool = False,
    inverse_preference: bool = False,
) -> np.ndarray:
    """
    Aggregate the inter-alternative statistics.

    :param first_alternatives: The first set of alternatives.
    :param second_alternatives: The second set of alternatives.
    :param feature_specs: The specification of the alternatives.
    :param stat_callback: The callback to calculate the statistics.
    :param average_out: Whether to average out the statistics using spec weigths, defaults to True.
    :param move_profiles: Whether to move the profiles, defaults to False.
    :param inverse_preference: Whether to inverse the preference, defaults to False.
    :return: The preferences.
    """
    aggregated_stat = np.zeros(
        (len(feature_specs), len(first_alternatives), len(second_alternatives))
    )
    for i, alternative in enumerate(first_alternatives):
        for j, other_alternative in enumerate(second_alternatives):
            if i == j and first_alternatives is second_alternatives:
                continue
            for k, (value, other_value, spec) in enumerate(
                zip(alternative, other_alternative, feature_specs)
            ):
                profile = j if move_profiles else 0
                diff = value - other_value
                diff *= -1 if spec.is_cost_type else 1
                diff *= -1 if inverse_preference else 1
                aggregated_stat[k, i, j] = stat_callback(diff, spec, profile)
    if not average_out:
        return aggregated_stat
    weigh_array = np.array([spec.weight for spec in feature_specs])
    broad_weigh_array = np.broadcast_to(
        weigh_array[:, np.newaxis, np.newaxis], aggregated_stat.shape
    )
    aggregated_stat = aggregated_stat * broad_weigh_array
    aggregated_stat = np.sum(aggregated_stat, axis=0)
    aggregated_stat /= weigh_array.sum()
    return aggregated_stat


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
) -> List[List[int]]:
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
) -> List[List[int]]:
    """
    Get the ordering of the alternatives according to the Promethee II algorithm.

    :param net_flow: The net flow.
    :return: The ordering of the alternatives.
    """
    net_flow = positive_flow - negative_flow
    sorted_net_desc_args = list(np.argsort(net_flow)[::-1])
    sorted_net_desc = np.sort(net_flow)[::-1]
    nodes = []
    for i, (net, arg) in enumerate(zip(sorted_net_desc, sorted_net_desc_args)):
        if i == 0 or net != sorted_net_desc[i - 1]:
            nodes.append([arg])
        else:
            nodes[-1].append(arg)
    return nodes


class PrometheeSolver:
    def __init__(
        self,
        alternatives: np.ndarray,
        feature_specs: List[FeatureSpec],
        promethe_one: bool = True,
    ) -> None:
        """
        :param alternatives: array of alternatives, where each row is an alternative and each column is a criterion.
        :param feature_specs: list of feature specs, one for each criterion.
        :param promethe_one: whether to use Promethee I or Promethee II ordering, defaults to True (Promethee I).
        """
        assert len(alternatives) > 0, "There must be at least one alternative."
        assert len(feature_specs) > 0, "There must be at least one criterion."
        assert (
            len(feature_specs) == alternatives.shape[1]
        ), "The number of criteria must match the number of columns in the alternatives matrix."
        self.alternatives = alternatives
        self.feature_specs = feature_specs
        self.order_func = (
            get_promethee_one_ordering if promethe_one else get_promethee_two_ordering
        )
        self.positive_flow = None
        self.negative_flow = None

    def solve(self) -> List[List[int]]:
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


def join_concordance_discordance(
    concordance: np.ndarray,
    inv_concordance: np.ndarray,
    discordance: np.ndarray,
    inv_discordance: np.ndarray,
) -> np.ndarray:
    """
    Join the concordance and discordance matrices.

    :param concordance: concordance matrix (a to b)
    :param inv_concordance: inverse concordance matrix (b to a)
    :param discordance: discordance matrix (a to b)
    :param inv_discordance: inverse discordance matrix (b to a)
    :return: joint concordance and discordance matrix
    """
    swapped_concordance = np.swapaxes(concordance[np.newaxis], 0, 2)
    swapped_discordance = np.swapaxes(discordance, 0, 2)
    swapped_inv_concordance = np.swapaxes(inv_concordance[np.newaxis], 0, 2)
    swapped_inv_discordance = np.swapaxes(inv_discordance, 0, 2)
    aggregated = np.concatenate([swapped_concordance, swapped_discordance], axis=2)
    aggregated_inv = np.concatenate(
        [swapped_inv_concordance, swapped_inv_discordance], axis=2
    )
    return np.concatenate([aggregated, aggregated_inv], axis=0)


def calc_sigma(entry: np.ndarray) -> float:
    """
    Calculate the sigma value for a single entry of concordance-dicordance matrix.
    (first entry is concordance, the rest are discordances)

    :param entry: entry of concordance-dicordance matrix
    :return: sigma value
    """
    concordance = entry[0]
    discordances = entry[1:][entry[1:] > concordance]
    return concordance * np.prod((1 - discordances) / (1 - concordance))


def determine_relation(entry: np.ndarray, threshold: float) -> int:
    """
    Determine the relation between two alternatives.
    Mapping is as follows:
    0: a is outranking b
    1: a and b are equivalent
    2: a and b are incomparable
    3: b is outranking a

    :param entry: entry of paired twofold matrix (a to b and b to a)
    :param threshold: threshold for the sigma value
    :return: relation between the alternatives
    """

    sigma, sigma_inv = entry
    outranking_forward = sigma >= threshold
    outranking_backward = sigma_inv >= threshold
    return (not outranking_forward) * 2 + outranking_backward


def to_classes(single_relations: np.ndarray, pessimistic: bool = False) -> np.ndarray:
    """
    Convert the single relations matrix to a matrix of classes.

    :param single_relations: single relations matrix
    :param pessimistic: whether to use pessimistic or optimistic classes
    :return: matrix of classes
    """
    cutoff = 2 if pessimistic else 3
    return (single_relations.T < cutoff).sum(axis=-1)


class ElectreTriBSolver:
    def __init__(
        self,
        alternatives: np.ndarray,
        boundary_profiles: np.ndarray,
        feature_specs: List[FeatureSpec],
        sigma_threshold: float,
        pessimistic_assigment: bool = False,
    ) -> None:
        """
        :param alternatives: array of alternatives, where each row is an alternative and each column is a criterion.
        :param boundary_profiles: array of boundary profiles, where each row is a boundary profile and each column is a criterion.
        :param feature_specs: list of feature specs, one for each criterion.
        :param sigma_threshold: threshold for the sigma value
        :param pessimistic_assigment: wheter to use pessimistic assignment, defaults to False
        """
        assert len(alternatives) > 0, "There must be at least one alternative."
        assert len(feature_specs) > 0, "There must be at least one criterion."
        assert (
            len(feature_specs) == alternatives.shape[1]
        ), "The number of criteria must match the number of columns in the alternatives matrix."
        assert (
            len(feature_specs) == boundary_profiles.shape[1]
        ), "The number of criteria must match the number of columns in the boundary profiles matrix."
        self.alternatives = alternatives
        self.boundary_profiles = boundary_profiles
        self.feature_specs = feature_specs
        self.sigma_threshold = sigma_threshold
        self.pessimistic_assigment = pessimistic_assigment

    @property
    def num_boundary_profiles(self) -> int:
        return self.boundary_profiles.shape[0]

    def solve(self) -> np.ndarray:
        """
        Solve the Electre Tri B problem.

        :return: vector of class assignments
        """
        params = [
            {
                "stat_callback": calculate_concordance,
                "average_out": True,
                "inverse_preference": False,
            },
            {
                "stat_callback": calculate_concordance,
                "average_out": True,
                "inverse_preference": True,
            },
            {
                "stat_callback": calculate_discordance,
                "average_out": False,
                "inverse_preference": False,
            },
            {
                "stat_callback": calculate_discordance,
                "average_out": False,
                "inverse_preference": True,
            },
        ]
        all_stats = [
            aggregate_inter_alternative_stats(
                self.alternatives,
                self.boundary_profiles,
                self.feature_specs,
                move_profiles=True,
                **param
            )
            for param in params
        ]
        concordance_discordance = join_concordance_discordance(*all_stats)
        sigma = np.apply_along_axis(calc_sigma, 2, concordance_discordance)[
            :, :, np.newaxis
        ]
        twofold_relations = np.dstack(
            [sigma[: self.num_boundary_profiles], sigma[self.num_boundary_profiles :]]
        )
        single_relations = np.apply_along_axis(
            partial(determine_relation, threshold=self.sigma_threshold),
            2,
            twofold_relations,
        )
        return to_classes(single_relations, pessimistic=self.pessimistic_assigment)


def calculate_additive_value_function(
    alternatives: np.ndarray, preference_matrix: np.ndarray
) -> np.ndarray:
    ...


def get_pref_matrix(ordering: List[List[int]]) -> np.ndarray:
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


def get_kendall_distance(
    ordering_1: list[list[int]], ordering_2: list[list[int]]
) -> float:
    """
    Get the Kendall distance between two orderings.

    :param ordering_1: ordering in the form of a list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
    :param ordering_2: ordering in the form of a list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
    :return: distance between the two orderings
    """
    pref_matrix_1 = get_pref_matrix(ordering_1)
    pref_matrix_2 = get_pref_matrix(ordering_2)
    return np.sum(np.abs(pref_matrix_1 - pref_matrix_2)) / 2


def get_kendall_tau(ordering_1: list[list[int]], ordering_2: list[list[int]]) -> float:
    """
    Get the Kendall tau between two orderings.

    :param ordering_1: ordering in the form of a list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
    :param ordering_2: ordering in the form of a list of preference nodes (e.g. [[0, 1], [2], [3]] for a 4-node graph),
    :return: kendall tau between the two orderings
    """
    m_size = max(max(node) for node in ordering_1) + 1
    distance = get_kendall_distance(ordering_1, ordering_2)
    return 1 - 4 * (distance / (m_size * (m_size - 1)))
