import numpy as np

from functools import partial
from typing import List

from mcda.types import FeatureSpec
from mcda.common import aggregate_inter_alternative_stats


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
    (first column is concordance, the rest are discordances)

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
