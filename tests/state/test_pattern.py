from math import factorial

import numpy as np
import pytest

from rubiks_cube.configuration.enumeration import Subset
from rubiks_cube.configuration.enumeration import Tag
from rubiks_cube.configuration.types import CubePattern
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.state.pattern import generate_pattern_symmetries_from_subset
from rubiks_cube.state.pattern import merge_patterns
from rubiks_cube.state.pattern import pattern_combinations
from rubiks_cube.tag import get_rubiks_cube_pattern
from rubiks_cube.tag.cubex import Cubex


class TestMergePatterns:
    def test_merge_patterns_single(self) -> None:
        patterns = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        ]
        merged = merge_patterns(patterns=patterns)
        assert np.array_equal(merged, np.array([1, 1, 0, 0, 0, 0, 0, 0]))

    def test_merge_patterns_duplicate(self) -> None:
        patterns = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 0, 0, 0]),
        ]
        merged = merge_patterns(patterns=patterns)
        assert np.array_equal(merged, np.array([1, 1, 0, 0, 0, 0, 0, 0]))

    def test_merge_patterns_disjoint(self) -> None:
        patterns = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 2, 2, 0, 0, 0, 0]),
        ]
        merged = merge_patterns(patterns=patterns)
        assert np.array_equal(merged, np.array([1, 1, 2, 2, 0, 0, 0, 0]))

    def test_merge_patterns_disjoint_same(self) -> None:
        patterns = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 1, 1, 0, 0, 0, 0]),
        ]
        merged = merge_patterns(patterns=patterns)
        assert np.array_equal(merged, np.array([1, 1, 2, 2, 0, 0, 0, 0]))

    def test_merge_patterns_overlap(self) -> None:
        patterns = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 1, 1, 0, 0, 0, 0, 0]),
        ]
        merged = merge_patterns(patterns=patterns)
        assert np.array_equal(merged, np.array([1, 2, 3, 0, 0, 0, 0, 0]))

    def test_merge_patterns_empty(self) -> None:
        patterns: list[CubePattern] = []
        with pytest.raises(ValueError):
            merge_patterns(patterns=patterns)

    def test_merge_patterns_unequal_len(self) -> None:
        patterns = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 1, 1, 0, 0, 0]),
        ]
        with pytest.raises(ValueError):
            merge_patterns(patterns=patterns)


class TestPatternCombinations:
    def test_pattern_combinations_solved(self) -> None:
        cube_size = 3
        pattern = get_rubiks_cube_pattern(tag="solved", cube_size=cube_size)
        n_combinations = pattern_combinations(pattern=pattern, cube_size=cube_size)
        assert n_combinations == 1

    def test_pattern_combinations_none(self) -> None:
        cube_size = 3
        pattern = get_rubiks_cube_pattern(tag="none", cube_size=cube_size)
        n_combinations = pattern_combinations(pattern=pattern, cube_size=cube_size)
        assert n_combinations == factorial(8) * 3**7 * factorial(12) * 2**11 / 2

    def test_pattern_combinations_eo(self) -> None:
        cube_size = 3
        pattern = get_rubiks_cube_pattern(tag="eo", subset="eo-fb", cube_size=cube_size)
        n_combinations = pattern_combinations(pattern=pattern, cube_size=cube_size)
        assert n_combinations == factorial(8) * 3**7 * factorial(12) / 2

    def test_pattern_combinations_dr(self) -> None:
        cube_size = 3
        pattern = get_rubiks_cube_pattern(tag="dr-ud", cube_size=cube_size)
        n_combinations = pattern_combinations(pattern=pattern, cube_size=cube_size)
        assert n_combinations == factorial(8) * factorial(8) * factorial(4) / 2

    def test_pattern_combinations_cross(self) -> None:
        cube_size = 3
        pattern = get_rubiks_cube_pattern(tag="cross", subset="cross-down", cube_size=cube_size)
        n_combinations = pattern_combinations(pattern=pattern, cube_size=cube_size)
        assert n_combinations == factorial(8) * 3**7 * factorial(8) * 2**7 / 2


class TestGeneratePatternsFromSubset:
    def test_generate_patterns_from_subset(self) -> None:
        cube_size = 3
        cubex = Cubex.from_settings(
            name=Tag.cross.value,
            solved_sequence=MoveSequence("R L U2 R2 L2 U2 R L U"),
            subset=Subset.down,
            cube_size=cube_size,
        )

        patterns, names = generate_pattern_symmetries_from_subset(
            pattern=cubex.patterns[0],
            initial_subset=Subset.down,
            prefix="cross",
            cube_size=cube_size,
        )

        assert len(patterns) == 6
        assert len(names) == 6
