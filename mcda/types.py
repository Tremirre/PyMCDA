import numpy as np

from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureSpec:
    name: str
    is_cost_type: bool
    indifference_thresholds: List[float]
    preference_thresholds: List[float]
    veto_thresholds: List[float] = field(default_factory=lambda: [np.inf])
    weight: float = 1.0


Ordering = List[List[int]]