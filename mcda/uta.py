import pulp
import numpy as np

from typing import List, Tuple

from mcda.common import map_to_unit_interval_2d
from mcda.order import get_ordering_from_value_array
from mcda.types import Ordering
from mcda.exceptions import FailedToSolveException, UTAInconsistencyException


class LinearUTASolver:
    def __init__(
        self,
        alternatives: np.ndarray,
        is_gain: np.ndarray,
        preferences: List[Tuple[int, int]],
        indifferences: List[Tuple[int, int]],
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
        self.shifted_alternatives -= 2 * is_gain * self.shifted_alternatives
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
        self.preferences = preferences
        self.indifferences = indifferences
        self.epsilon = espilon
        self.problem = None  # pulp.LpProblem("UTA", pulp.LpMinimize)
        bin_vars_names = [(i, j) for i, j in preferences + indifferences]
        self.bin_vars = {
            (i, j): pulp.LpVariable(f"V{i}{j}", cat="Binary") for i, j in bin_vars_names
        }
        self.weights = pulp.LpVariable.dicts(
            "w", range(alternatives.shape[1]), lowBound=0, upBound=1
        )
        self.all_inconsistent_constraints = []

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
            value = sum(
                alternative[k] * self.weights[k].varValue
                for k in range(self.shifted_alternatives.shape[1])
            )
            values.append(value)
        return np.array(values)

    @property
    def attribute_linear_function_params(self) -> dict[str, np.ndarray]:
        """
        Returns the linear function parameters for each attribute.

        :raises RuntimeError: when the solver has not been solved yet
        :return: dictionary with keys "slopes" and "horizontal_intercepts" containing the linear function parameters
        """
        if self.problem.status != 1:
            raise RuntimeError("Solver must be solved before accessing the results")
        linear_coeff = (
            self.alternative_values
            / (self.alternatives.max(axis=0) - self.alternatives.min(axis=0))
        ) * (self.is_gain - 1 * (1 - self.is_gain))
        standalone_coeff = self.alternatives.min(
            axis=0
        ) * self.is_gain + self.alternatives.max(axis=0) * (1 - self.is_gain)
        return {"slopes": linear_coeff, "horizontal_intercepts": standalone_coeff}

    def _alternative_constraints(self, idx: int) -> pulp.LpAffineExpression:
        """
        Returns the additive value function for the alternative with index `idx`.

        :param idx: index of the alternative
        :return: additive value function for the alternative with index `idx`
        """
        return sum(
            self.shifted_alternatives[idx, k] * self.weights[k]
            for k in range(self.alternatives.shape[1])
        )

    def _reset_state(self) -> None:
        """
        Resets the solver state (i.e. resets all constraints and variables).
        """
        self.problem = pulp.LpProblem("UTA", pulp.LpMinimize)
        self.problem += pulp.lpSum(self.bin_vars.values())
        self.problem += sum(self.weights.values()) == 1
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
