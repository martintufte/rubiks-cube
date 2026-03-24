from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from rubiks_cube.configuration.types import BoolArray
    from rubiks_cube.configuration.types import CubePattern
    from rubiks_cube.configuration.types import CubePermutation


@attrs.mutable
class SearchProblem:
    actions: dict[str, CubePermutation]
    pattern: CubePattern
    adj_matrix: BoolArray | None = None


@attrs.mutable
class Transform(ABC):

    @abstractmethod
    def fit(self, search_problem: SearchProblem) -> SearchProblem:
        """Fit self to the state."""

    @abstractmethod
    def transform_permutation(self, permutation: CubePermutation) -> CubePermutation:
        """Transform the permutation."""
