import pulp
import numpy as np

from typing import List, Tuple, Union

from pymcda.common import map_to_unit_interval_2d
from pymcda.order import get_ordering_from_value_array
from pymcda.types import Ordering
from pymcda.exceptions import FailedToSolveException, UTAInconsistencyException


def piecewise_utility_function(
    x: float, weights: List[Union[pulp.LpVariable, float]]
) -> Union[pulp.LpAffineExpression, float]:
    """
    Creates a piecewise utility function for the given value of the criterion.
    Assumes number of pieces equal to the number of weights.

    :param x: value of the criterion
    :param weights: list of weights for the criteria, where each weight is a float or a pulp.LpVariable
    :return: piecewise utility function for the given value of the criterion (or the value itself if weights are concrete values)
    """
    if not weights:
        raise ValueError("Weights must not be empty")
    if len(weights) == 1:
        return weights[0] * x
    piece_length = 1 / len(weights)
    x_tick = int(x // piece_length)
    base = piece_length * sum(weights[j] for j in range(x_tick))
    if x_tick == len(weights):
        return base
    fraction = (x % piece_length) * weights[x_tick]
    return base + fraction


class UTASolver:
    def __init__(
        self,
        alternatives: np.ndarray,
        is_gain: np.ndarray,
        preferences: List[Tuple[int, int]],
        indifferences: List[Tuple[int, int]],
        num_pieces: int = 1,
        espilon: float = 1e-6,
    ) -> None:
        """
        :param alternatives: array of alternatives, where each row is an alternative and each column is a criterion.
        :param is_gain: array of booleans indicating whether the criterion is a gain criterion or not.
        :param preferences: list of tuples of indices of alternatives that are preferred over each other
            (i.e. (i, j) means that alternative i is preferred over alternative j)
        :param indifferences: list of tuples of indices of alternatives that are indifferent to each other
            (i.e. (i, j) means that alternative i is indifferent to alternative j)
        :param espilon: used in representing strict inequality constraints, defaults to 1e-6
        :raises ValueError: when the preference relation contains self-preferences
        :raises ValueError: when the preference and indifference relations are not disjoint
        """
        self.alternatives = alternatives
        self.is_gain = is_gain
        self.shifted_alternatives = map_to_unit_interval_2d(alternatives)
        self.shifted_alternatives[:, ~is_gain] = (
            1 - self.shifted_alternatives[:, ~is_gain]
        )

        for p_i, p_j in preferences:
            if p_i == p_j:
                raise ValueError(
                    "Preference relation must not contain self-preferences"
                )
            for i_i, i_j in indifferences:
                if sorted([p_i, p_j]) == sorted([i_i, i_j]):
                    raise ValueError(
                        f"Preference and indifference relations must be disjoint - ({p_i}, {p_j}) and ({i_i}, {i_j})"
                    )
        if num_pieces < 1:
            raise ValueError("Number of pieces must be at least 1")
        self.num_pieces = num_pieces
        self.preferences = preferences
        self.indifferences = indifferences
        self.epsilon = espilon
        self.problem = None
        bin_vars_names = [(i, j) for i, j in preferences + indifferences]
        self.bin_vars = {
            (i, j): pulp.LpVariable(f"V{i}{j}", cat="Binary") for i, j in bin_vars_names
        }
        self.weights = [
            [
                pulp.LpVariable(f"w{i}{j}", lowBound=0, upBound=1)
                for j in range(num_pieces)
            ]
            for i in range(alternatives.shape[1])
        ]
        self.all_inconsistent_constraints = []

    def utility_functions_as_sample_points(self, sample_size: int = 100) -> np.ndarray:
        """
        Returns the utility functions as sample points.

        :raises RuntimeError: when the solver has not been solved yet
        :return: matrix of utility functions as sample points,
            where each row is a utility function and each column is a sample point
        """
        if self.problem.status != 1:
            raise RuntimeError("Solver must be solved before accessing the results")
        space = np.linspace(0, 1, sample_size)
        utility_functions = []
        for sub_weights in self.weights:
            utility_functions.append(
                np.array(
                    [
                        piecewise_utility_function(v, [w.value() for w in sub_weights])
                        for v in space
                    ]
                )
            )
        return np.array(utility_functions)

    @property
    def alternative_values(self) -> np.ndarray:
        """
        Returns the values of the additive value function for all alternatives.

        :raises RuntimeError: when the solver has not been solved yet
        :return: array of values of the additive value function for all alternatives
        """
        if self.problem.status != 1:
            raise RuntimeError("Solver must be solved before accessing the results")
        values = []
        for alternative in self.shifted_alternatives:
            values.append(
                sum(
                    piecewise_utility_function(
                        attr_value, [w.value() for w in self.weights[attr_idx]]
                    )
                    for attr_idx, attr_value in enumerate(alternative)
                )
            )
        return np.array(values)

    def _alternative_constraints(self, idx: int) -> pulp.LpAffineExpression:
        """
        Returns the additive value function for the alternative with index `idx`.

        :param idx: index of the alternative
        :return: additive value function for the alternative with index `idx`
        """
        return sum(
            piecewise_utility_function(attr_value, self.weights[attr_idx])
            for attr_idx, attr_value in enumerate(self.shifted_alternatives[idx])
        )

    def _reset_state(self) -> None:
        """
        Resets the solver state (i.e. resets all constraints and variables).
        """
        self.problem = pulp.LpProblem("UTA", pulp.LpMinimize)
        self.problem += pulp.lpSum(self.bin_vars.values())
        self.problem += (
            sum(sum(subweights) for subweights in self.weights) == self.num_pieces
        )
        self.all_inconsistent_constraints.clear()

    def _find_all_inconsistent_preferences(self) -> None:
        """
        Iteratively finds all inconsistent constraint sets.
        """
        inconsistent_constraints = [
            bin_var_tag
            for bin_var_tag, bin_var in self.bin_vars.items()
            if bin_var.varValue == 1
        ]
        if not inconsistent_constraints:
            return
        self.all_inconsistent_constraints.append(inconsistent_constraints)
        while True:
            self.problem += (
                pulp.lpSum(self.bin_vars[v] for v in inconsistent_constraints)
                <= len(inconsistent_constraints) - 1
            )
            self.problem.solve()
            if self.problem.status != 1:
                return
            inconsistent_constraints = [
                bin_var_tag
                for bin_var_tag, bin_var in self.bin_vars.items()
                if bin_var.varValue == 1
            ]
            if inconsistent_constraints:
                self.all_inconsistent_constraints.append(inconsistent_constraints)

    def solve(self) -> Ordering:
        """
        Solve the UTA problem.
        Resets the solver state (i.e. resets all constraints and variables) before solving.
        In case of inconsistency, aggregates all inconsistent constraint sets.

        :raises UTAInconsistencyException: when the provided preferences and indifferences are inconsistent
        :raises FailedToSolveException: when the solver fails to find a solution
        :return: ordering of alternatives according to obtained additive value function
        """
        self._reset_state()
        for i, j in self.preferences:
            lhs = self._alternative_constraints(i)
            rhs = self._alternative_constraints(j)
            self.problem += lhs - self.epsilon >= rhs - self.bin_vars[(i, j)]

        for i, j in self.indifferences:
            lhs = self._alternative_constraints(i)
            rhs = self._alternative_constraints(j)

            self.problem += lhs >= rhs - self.bin_vars[(i, j)]
            self.problem += rhs >= lhs - self.bin_vars[(i, j)]

        self.problem.solve()
        self._find_all_inconsistent_preferences()
        if self.all_inconsistent_constraints:
            raise UTAInconsistencyException(self.all_inconsistent_constraints)
        if self.problem.status != 1:
            raise FailedToSolveException()

        values = self.alternative_values
        ordering = get_ordering_from_value_array(values, ascending=False)
        return ordering
