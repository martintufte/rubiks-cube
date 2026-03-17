from __future__ import annotations

import logging
import timeit
from functools import lru_cache
from math import log2
from threading import Lock
from typing import TYPE_CHECKING
from typing import Any
from typing import Self  # ty: ignore[unresolved-import]
from typing import Sequence

import attrs
import numpy as np

from rubiks_cube.autotagger.subset import distinguish_htr
from rubiks_cube.configuration.enumeration import Goal
from rubiks_cube.configuration.enumeration import Piece
from rubiks_cube.configuration.enumeration import Symmetry
from rubiks_cube.move.generator import MoveGenerator
from rubiks_cube.move.meta import MoveMeta
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.representation.mask import get_piece_mask
from rubiks_cube.representation.mask import get_rubiks_cube_mask
from rubiks_cube.representation.pattern import generate_pattern_symmetries_variations
from rubiks_cube.representation.pattern import get_empty_pattern
from rubiks_cube.representation.pattern import get_identity_pattern
from rubiks_cube.representation.pattern import merge_patterns
from rubiks_cube.representation.pattern import pattern_combinations
from rubiks_cube.representation.pattern import pattern_from_generator
from rubiks_cube.representation.pattern import pattern_implies

if TYPE_CHECKING:
    from rubiks_cube.configuration.types import CubePattern
    from rubiks_cube.configuration.types import CubePermutation
    from rubiks_cube.configuration.types import PermutationValidator


LOGGER = logging.getLogger(__name__)
GET_PATTERNS_LOCK = Lock()


@attrs.mutable
class Pattern:
    variations: dict[Symmetry, CubePattern]
    validator: PermutationValidator | None = None

    @classmethod
    def from_settings(
        cls,
        move_meta: MoveMeta,
        symmetry: Symmetry,
        solved_sequence: MoveSequence | None = None,
        pieces: list[Piece] | None = None,
        orientation_generator: MoveGenerator | None = None,
    ) -> Self:
        """Create Pattern from symmetry, a solved sequence and orientations.

        Args:
            move_meta (MoveMeta): Meta information about moves.
            symmetry (Symmetry, optional): Symmetry for creating variations.
                Defaults to Symmetry.none.
            solved_sequence (MoveSequence, optional): Sequence that only fixate the pattern.
            pieces (list[Piece], optional): List of pieces.
            orientation_generator (MoveGenerator, optional): Generators conserving orientations.

        Returns:
            Self: Patterns.
        """
        if solved_sequence is None:
            fixate_pattern = get_empty_pattern(cube_size=move_meta.cube_size)
        else:
            fixate_mask = get_rubiks_cube_mask(sequence=solved_sequence, move_meta=move_meta)
            fixate_pattern = get_identity_pattern(cube_size=move_meta.cube_size)
            fixate_pattern[~fixate_mask] = 0

        # Find the orientations of the pieces given the generator
        if pieces is not None and orientation_generator is not None:
            mask = get_piece_mask(piece=pieces, cube_size=move_meta.cube_size)
            orientations_pattern = pattern_from_generator(
                generator=orientation_generator,
                mask=mask,
                move_meta=move_meta,
            )
            # Don't keep elements that appear only once
            unique, counts = np.unique(orientations_pattern, return_counts=True)
            mask = np.isin(orientations_pattern, unique[counts == 1])
            orientations_pattern[mask] = 0
        else:
            orientations_pattern = get_empty_pattern(cube_size=move_meta.cube_size)

        # Store all variations
        variations = {symmetry: merge_patterns((fixate_pattern, orientations_pattern))}

        if symmetry is not Symmetry.none:
            initial_symmetry, initial_pattern = next(iter(variations.items()))
            variations = generate_pattern_symmetries_variations(
                pattern=initial_pattern,
                symmetry=initial_symmetry,
                move_meta=move_meta,
            )
        return cls(variations=variations)

    @classmethod
    def from_merge(
        cls,
        move_meta: MoveMeta,
        symmetry: Symmetry,
        patterns: Sequence[CubePattern],
    ) -> Self:
        variations = {symmetry: merge_patterns(patterns=patterns)}
        if symmetry is not Symmetry.none:
            initial_symmetry, initial_pattern = next(iter(variations.items()))
            variations = generate_pattern_symmetries_variations(
                pattern=initial_pattern,
                symmetry=initial_symmetry,
                move_meta=move_meta,
            )
        return cls(variations=variations)

    def __and__(self, other: Pattern) -> Pattern:
        return Pattern(
            variations={
                symmetry: merge_patterns((pattern, other_pattern))
                for symmetry, pattern in self.variations.items()
                for _other_symmetry, other_pattern in other.variations.items()
            },
        )

    def __getitem__(self, key: Symmetry) -> CubePattern:
        return self.variations[key]

    def __contains__(self, other: Any) -> bool:
        if isinstance(other, Pattern):
            return any(
                pattern_implies(variation, other_variation)
                for variation in self.variations.values()
                for other_variation in other.variations.values()
            )
        return False

    def __len__(self) -> int:
        return len(self.variations)

    def match(self, permutation: CubePermutation) -> bool:
        if any(
            np.array_equal(variation[permutation], variation)
            for variation in self.variations.values()
        ):
            if self.validator is not None:
                return self.validator(permutation)
            return True
        return False

    def calc_combinations(self, move_meta: MoveMeta) -> int:
        """Sum of the number of combinations for each pattern."""
        initial_pattern = next(iter(self.variations.values()))
        n = len(self) * pattern_combinations(initial_pattern, move_meta)

        # TODO: Fix hack for counting with validator. Right now, only htr has a validator
        if self.validator is not None:
            return n // 6
        return n

    def entropy(self, move_meta: MoveMeta) -> float:
        """Find the estimated entropy of the patterns."""
        return log2(self.calc_combinations(move_meta=move_meta))


def get_3x3_patterns(move_meta: MoveMeta) -> dict[Goal, Pattern]:
    patterns: dict[Goal, Pattern] = {}

    # Only fixate
    fixate_goals: dict[Goal, tuple[list[str], Symmetry]] = {
        Goal.none: (["x", "y"], Symmetry.none),
        Goal.cp_layer: (["M'", "S", "Dw"], Symmetry.up),
        Goal.ep_layer: (["M2", "D2", "F2", "B2", "Dw"], Symmetry.up),
        Goal.layer: (["Dw"], Symmetry.up),
        Goal.cross: (["R", "L", "U2", "R2", "L2", "U2", "R", "L", "U"], Symmetry.down),
        Goal.f2l: (["U"], Symmetry.down),
        Goal.x_cross: (["R", "L'", "U2", "R2", "L", "U2", "R", "U"], Symmetry.down_bl),
        Goal.xx_cross_adjacent: (["R", "L'", "U2", "R'", "L", "U"], Symmetry.down_b),
        Goal.xx_cross_diagonal: (["R'", "L'", "U2", "R", "L", "U"], Symmetry.down_bl_fr),
        Goal.xxx_cross: (["R", "U", "R'", "U"], Symmetry.down_fr),
        Goal.block_1x1x3: (["Fw", "Rw"], Symmetry.bl),
        Goal.block_1x2x2: (["U", "R", "Fw"], Symmetry.back_dl),
        Goal.block_1x2x3: (["U", "Rw"], Symmetry.dl),
        Goal.block_2x2x2: (["U", "R", "F"], Symmetry.down_bl),
        Goal.block_2x2x3: (["U", "R"], Symmetry.dl),
        Goal.corners: (["M'", "S", "E"], Symmetry.none),
        Goal.edges: (
            ["E2", "R", "L", "S2", "L", "R'", "S2", "R2", "S", "M", "S", "M'"],
            Symmetry.none,
        ),
        Goal.solved: ([], Symmetry.none),
        Goal.minus_slice: (["M"], Symmetry.m),
    }
    for goal, (moves, symmetry) in fixate_goals.items():
        patterns[goal] = Pattern.from_settings(
            move_meta=move_meta,
            symmetry=symmetry,
            solved_sequence=MoveSequence(moves),
        )

    # Orientations
    patterns[Goal.co_face] = Pattern.from_settings(
        move_meta=move_meta,
        symmetry=Symmetry.up,
        solved_sequence=MoveSequence(["y"]),
        pieces=[Piece.corner],
        orientation_generator=MoveGenerator.from_str("<U>"),
    )
    patterns[Goal.eo_face] = Pattern.from_settings(
        move_meta=move_meta,
        solved_sequence=MoveSequence(["y"]),
        pieces=[Piece.edge],
        orientation_generator=MoveGenerator.from_str("<U>"),
        symmetry=Symmetry.up,
    )
    patterns[Goal.face] = Pattern.from_settings(
        move_meta=move_meta,
        solved_sequence=MoveSequence(["y"]),
        pieces=[Piece.corner, Piece.edge],
        orientation_generator=MoveGenerator.from_str("<U>"),
        symmetry=Symmetry.up,
    )

    # Symmetric edge orientations
    edge_orientation_symmetric = {
        Goal.eo: ("<F2, B2, L, R, U, D>", Symmetry.fb),
        Goal.eo_floppy: ("<L2, R2, U2, D2>", Symmetry.fb),
    }
    for goal, (generator, symmetry) in edge_orientation_symmetric.items():
        patterns[goal] = Pattern.from_settings(
            move_meta=move_meta,
            pieces=[Piece.edge],
            orientation_generator=MoveGenerator.from_str(generator),
            symmetry=symmetry,
        )

    # Symmetric corner orientations
    corner_orientation_symmetric = {
        Goal.co: ("<F, B, L2, R2, U2, D2>", Symmetry.fb),
    }
    for goal, (generator, symmetry) in corner_orientation_symmetric.items():
        patterns[goal] = Pattern.from_settings(
            move_meta=move_meta,
            symmetry=symmetry,
            pieces=[Piece.corner],
            orientation_generator=MoveGenerator.from_str(generator),
        )

    # Symmetric center orientations
    patterns[Goal.xo] = Pattern.from_settings(
        move_meta=move_meta,
        symmetry=Symmetry.fb,
        pieces=[Piece.center],
        orientation_generator=MoveGenerator.from_str("<x2, z>"),
    )

    # Symmetric composite
    patterns[Goal.f2l_face] = patterns[Goal.face] & patterns[Goal.f2l]
    patterns[Goal.f2l_co] = patterns[Goal.co_face] & patterns[Goal.f2l]
    patterns[Goal.f2l_eo] = patterns[Goal.eo_face] & patterns[Goal.f2l]
    patterns[Goal.f2l_cp] = patterns[Goal.cp_layer] & patterns[Goal.f2l]
    patterns[Goal.f2l_ep] = patterns[Goal.ep_layer] & patterns[Goal.f2l]
    patterns[Goal.f2l_ep_co] = patterns[Goal.f2l_co] & patterns[Goal.ep_layer]
    patterns[Goal.f2l_eo_cp] = patterns[Goal.f2l_cp] & patterns[Goal.eo_face]

    # Non-symmetric edge orientations
    edge_orientation_tags = {
        Goal.eo_htr: "<F2, B2, L2, R2, U2, D2>",
    }
    for goal, generator in edge_orientation_tags.items():
        patterns[goal] = Pattern.from_settings(
            move_meta=move_meta,
            symmetry=Symmetry.none,
            pieces=[Piece.edge],
            orientation_generator=MoveGenerator.from_str(generator),
        )

    # Non-symmetric corner orientations
    corner_orientation_tags = {
        Goal.co_htr: "<F2, B2, L2, R2, U2, D2>",
    }
    for goal, generator in corner_orientation_tags.items():
        patterns[goal] = Pattern.from_settings(
            move_meta=move_meta,
            symmetry=Symmetry.none,
            pieces=[Piece.corner],
            orientation_generator=MoveGenerator.from_str(generator),
        )

    # Composite patterns
    patterns[Goal.xo_htr] = Pattern.from_merge(
        move_meta=move_meta,
        symmetry=Symmetry.none,
        patterns=list(patterns[Goal.xo].variations.values()),
    )

    patterns[Goal.dr] = Pattern.from_merge(
        move_meta=move_meta,
        symmetry=Symmetry.ud,
        patterns=[
            patterns[Goal.co][Symmetry.ud],
            patterns[Goal.eo][Symmetry.fb],
            patterns[Goal.eo][Symmetry.lr],
            patterns[Goal.xo_htr][Symmetry.none],
        ],
    )

    patterns[Goal.leave_slice] = Pattern.from_merge(
        move_meta=move_meta,
        symmetry=Symmetry.e,
        patterns=[
            patterns[Goal.minus_slice][Symmetry.e],
            patterns[Goal.eo][Symmetry.fb],
            patterns[Goal.eo][Symmetry.lr],
            patterns[Goal.xo][Symmetry.fb],
            patterns[Goal.xo][Symmetry.lr],
        ],
    )

    patterns[Goal.htr_like] = patterns[Goal.co_htr] & patterns[Goal.eo_htr] & patterns[Goal.xo_htr]

    # Add real htr with permutation validator
    htr_like = patterns[Goal.htr_like]
    patterns[Goal.htr] = Pattern(
        variations=htr_like.variations,
        validator=lambda permutation: distinguish_htr(permutation) == "real",
    )

    # TODO(martin): Consider doing composite patterns Goal.xx_cross (adj or diag)

    return patterns


def sort_using_entropy(patterns: dict[Goal, Pattern], move_meta: MoveMeta) -> dict[Goal, Pattern]:
    """Sort the patterns using a proxy for the entropy.

    For future, consider an anytime iterative inclusion/exclusion sorting algorithm.
    """

    def entropy(goal: Goal) -> float:
        return patterns[goal].entropy(move_meta=move_meta)

    return {goal: patterns[goal] for goal in sorted(patterns, key=entropy)}


@lru_cache(maxsize=10)
def _get_cached_patterns(cube_size: int) -> dict[Goal, Pattern]:
    """Return a cached dictionary of patterns from goals given the cube size."""
    assert cube_size == 3

    move_meta = MoveMeta.from_cube_size(cube_size)

    t = timeit.default_timer()
    patterns = get_3x3_patterns(move_meta=move_meta)

    LOGGER.debug(f"Created patterns in {timeit.default_timer() - t:.3f} seconds.")

    if cube_size < 4:
        t = timeit.default_timer()
        patterns = sort_using_entropy(patterns, move_meta=move_meta)
        LOGGER.debug(f"Sorted patterns in {timeit.default_timer() - t:.3f} seconds.")

    return patterns


def get_patterns(cube_size: int) -> dict[Goal, Pattern]:
    """Return a dictionary of patterns given the cube size.

    Args:
        cube_size (int): Size of the cube.

    Returns:
        dict[Goal, Pattern]: Dictionary of goals and their patterns.

    Note:
        - Caches patterns with single-flight initialization per process.
          `functools.lru_cache` can execute the wrapped function
          more than once when concurrent cold calls happen.
    """
    with GET_PATTERNS_LOCK:
        return _get_cached_patterns(cube_size)
