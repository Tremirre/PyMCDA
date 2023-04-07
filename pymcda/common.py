import typing

import numpy as np

from pymcda.types import FeatureSpec


def aggregate_inter_alternative_stats(
    first_alternatives: np.ndarray,
    second_alternatives: np.ndarray,
    feature_specs: typing.List[FeatureSpec],
    stat_callback: typing.Callable[[float, FeatureSpec, int], float],
    average_out: bool = True,
    move_profiles: bool = False,
    inverse_preference: bool = False,
) -> np.ndarray:
    """
    Aggregate the inter-alternative statistics.

    :param first_alternatives: The first array of alternatives.
    :param second_alternatives: The second array of alternatives.
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


def map_to_unit_interval_2d(x: np.ndarray) -> np.ndarray:
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
