from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from rubiks_cube.group.so3 import DEFAULT_STATE_ORDER
from rubiks_cube.group.so3 import get_canonical_sequence
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.representation import get_rubiks_cube_permutation

if TYPE_CHECKING:
    from rubiks_cube.configuration.types import CubeMask
    from rubiks_cube.configuration.types import CubePermutation


LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=10)
def _standard_rotation_permutations(cube_size: int) -> tuple[CubePermutation, ...]:
    return tuple(
        get_rubiks_cube_permutation(
            sequence=MoveSequence(list(get_canonical_sequence(state))),
            cube_size=cube_size,
        )
        for state in DEFAULT_STATE_ORDER
    )


def find_rotation_offset(
    permutation: CubePermutation,
    affected_mask: CubeMask | None,
    cube_size: int,
) -> CubePermutation | None:
    """Find the rotational offset between the permutation and the mask.

    It finds the rotation such that permutation[~affected_mask] == identity[~affected_mask].

    Args:
        permutation (CubePermutation): Initial permutation.
        affected_mask (CubeMask | None, optional): Mask of what is affected by the actions.
        cube_size (int): Size of the cube.

    Returns:
        CubePermutation | None: Rotation offset for the permutation.

    Raises:
        ValueError: could not infer cube size.
    """

    # Assume everything is affected if not provided
    if affected_mask is None:
        affected_mask = np.ones_like(permutation, dtype=bool)

    not_affected_mask = ~affected_mask

    for rotation in _standard_rotation_permutations(cube_size):
        if np.array_equal(rotation[not_affected_mask], permutation[not_affected_mask]):
            return rotation

    return None
