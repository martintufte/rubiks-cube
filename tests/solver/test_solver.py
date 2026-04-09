from __future__ import annotations

import numpy as np

from rubiks_cube.autotagger.pattern import get_patterns
from rubiks_cube.configuration import DEFAULT_GENERATOR_MAP
from rubiks_cube.configuration.enumeration import Goal
from rubiks_cube.configuration.enumeration import SearchSide
from rubiks_cube.configuration.enumeration import SolveStrategy
from rubiks_cube.configuration.enumeration import Status
from rubiks_cube.configuration.enumeration import Variant
from rubiks_cube.move.generator import MoveGenerator
from rubiks_cube.move.meta import MoveMeta
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.representation import get_rubiks_cube_permutation
from rubiks_cube.solver import solve_pattern
from rubiks_cube.solver.actions import get_actions
from rubiks_cube.solver.bidirectional import BidirectionalSolver


def test_main() -> None:
    """Example of solving a step with a generator on a 3x3 cube."""
    move_meta = MoveMeta.from_cube_size(3)

    sequence = MoveSequence.from_str("M2 U M U2 M' U M2")
    generator = MoveGenerator.from_str("<M, U>")

    search_summary = solve_pattern(
        sequence=sequence,
        move_meta=move_meta,
        generator=generator,
        algorithms=None,
        goal=Goal.solved,
        max_search_depth=8,
        max_solutions=1,
        solve_strategy=SolveStrategy.normal,
    )
    solutions = search_summary.solutions

    assert isinstance(solutions, list)
    assert len(solutions) == 1
    assert search_summary.walltime > 0
    assert search_summary.status is Status.Success


def test_default() -> None:
    """Example of solving a step with a generator on a 3x3 cube."""
    move_meta = MoveMeta.from_cube_size(3)

    scrambles = [
        MoveSequence.from_str("L"),
        MoveSequence.from_str("R"),
        MoveSequence.from_str("U"),
        MoveSequence.from_str("D"),
        MoveSequence.from_str("F"),
        MoveSequence.from_str("B"),
    ]
    generator = MoveGenerator.from_str(DEFAULT_GENERATOR_MAP[3])

    for scramble in scrambles:
        search_summary = solve_pattern(
            sequence=scramble,
            move_meta=move_meta,
            generator=generator,
            algorithms=None,
            goal=Goal.solved,
            max_search_depth=10,
            max_solutions=2,
            solve_strategy=SolveStrategy.normal,
        )
        solutions = search_summary.solutions
        assert len(solutions) == 2
        assert isinstance(solutions, list)
        assert search_summary.walltime > 0
        assert search_summary.status is Status.Success

        # First solution has length == 1
        assert len(solutions[0]) == 1
        # Second solution has length == 8
        assert len(solutions[1]) == 8


def test_search_inverse() -> None:
    scramble = MoveSequence.from_str("R")
    generator = MoveGenerator.from_str(DEFAULT_GENERATOR_MAP[3])
    move_meta = MoveMeta.from_cube_size(3)

    search_summary = solve_pattern(
        sequence=scramble,
        move_meta=move_meta,
        generator=generator,
        algorithms=None,
        goal=Goal.solved,
        max_search_depth=10,
        max_solutions=1,
        solve_strategy=SolveStrategy.inverse,
    )

    assert search_summary.status is Status.Success
    assert len(search_summary.solutions) == 1
    assert len(search_summary.solutions[0]) == 1
    assert len(search_summary.solutions[0].inverse) > 0


def test_solve_from_rotated_state() -> None:
    """Solve from a rotated state: solutions should be expressed in the rotated frame."""
    move_meta = MoveMeta.from_cube_size(3)
    generator = MoveGenerator.from_str(DEFAULT_GENERATOR_MAP[3])

    # y R: rotate cube with y, then turn right face in y-rotated orientation.
    # Equivalent in original frame: conjugated_R y (via shift_rotations_to_end).
    scramble_rotated = MoveSequence.from_str("y R")

    summary_rotated = solve_pattern(
        sequence=scramble_rotated,
        move_meta=move_meta,
        generator=generator,
        goal=Goal.solved,
        max_search_depth=4,
        max_solutions=1,
        solve_strategy=SolveStrategy.normal,
    )

    assert summary_rotated.status is Status.Success
    assert len(summary_rotated.solutions) == 1
    solution = summary_rotated.solutions[0]

    # The solution should be length 1 (undo a single-move scramble)
    assert len(solution) == 1

    # Verify the solution is expressed in the rotated frame: applying the solution in
    # the y-rotated frame (= conjugate(move, y) in original frame) after the scramble's
    # de-oriented state restores the cube.
    de_oriented_perm = get_rubiks_cube_permutation(
        sequence=scramble_rotated,
        move_meta=move_meta,
        orientate_after=True,
    )
    # Each solution move, when expressed back in original frame via y-conjugation, undoes
    # the de-oriented scramble.
    for move in solution.normal:
        conjugated = move_meta.conjugation_map.get((move, "y"), move)
        conjugated_perm = move_meta.permutations[conjugated]
        result = de_oriented_perm[conjugated_perm]
        identity = np.arange(move_meta.size, dtype=move_meta.dtype)
        assert np.array_equal(
            result, identity
        ), f"Solution move '{move}' (conjugated='{conjugated}') does not solve the de-oriented state"


def test_solve_rotated_same_length_as_deoriented() -> None:
    """Solve 'y R' and the equivalent de-oriented scramble: solutions should have equal length.

    y R == conjugation_map[(R, y)] y. Solving the de-oriented state finds a 1-move solution.
    Solving from the y-rotated state should also produce a 1-move solution (just in a different
    basis), since the scramble complexity is the same.
    """
    move_meta = MoveMeta.from_cube_size(3)
    generator = MoveGenerator.from_str(DEFAULT_GENERATOR_MAP[3])

    # Equivalent de-oriented scramble: conjugate R by y to get the original-frame equivalent
    deoriented_move = move_meta.conjugation_map.get(("R", "y"), "R")
    scramble_deoriented = MoveSequence.from_str(deoriented_move)
    scramble_rotated = MoveSequence.from_str("y R")

    summary_deoriented = solve_pattern(
        sequence=scramble_deoriented,
        move_meta=move_meta,
        generator=generator,
        goal=Goal.solved,
        max_search_depth=4,
        max_solutions=1,
        solve_strategy=SolveStrategy.normal,
    )
    summary_rotated = solve_pattern(
        sequence=scramble_rotated,
        move_meta=move_meta,
        generator=generator,
        goal=Goal.solved,
        max_search_depth=4,
        max_solutions=1,
        solve_strategy=SolveStrategy.normal,
    )

    assert summary_deoriented.status is Status.Success
    assert summary_rotated.status is Status.Success
    assert len(summary_deoriented.solutions[0]) == len(summary_rotated.solutions[0])


def test_solve_non_canonical_variant_solution_is_valid() -> None:
    """Solving for a non-canonical variant via canonical frame conjugation gives a valid solution.

    EO.fb is the initial pattern variant; its canonical (Variant.ud) uses rotation ["x"].
    Scramble F breaks EO.fb (F is a "bad" move for the fb axis). The solver should find
    a 1-move solution, and applying that solution after the scramble must achieve EO.fb.
    """
    move_meta = MoveMeta.from_cube_size(3)
    generator = MoveGenerator.from_str(DEFAULT_GENERATOR_MAP[3])
    patterns = get_patterns(cube_size=3)
    eo_pattern = patterns[Goal.eo]

    scramble = MoveSequence.from_str("F")

    summary = solve_pattern(
        sequence=scramble,
        move_meta=move_meta,
        generator=generator,
        goal=Goal.eo,
        variants=[Variant.fb],
        max_search_depth=4,
        max_solutions=1,
        solve_strategy=SolveStrategy.normal,
    )

    assert summary.status is Status.Success
    assert len(summary.solutions) == 1
    solution = summary.solutions[0]
    assert len(solution) == 1

    # Verify: applying the solution after the scramble achieves EO.fb.
    scramble_perm = get_rubiks_cube_permutation(sequence=scramble, move_meta=move_meta)
    solution_perm = get_rubiks_cube_permutation(sequence=solution, move_meta=move_meta)
    final_perm = scramble_perm[solution_perm]
    assert eo_pattern.match(final_perm) is not None


def test_bidirectional_solver_search_many_returns_rooted_solutions() -> None:
    move_meta = MoveMeta.from_cube_size(3)

    actions = get_actions(move_meta=move_meta, generator=MoveGenerator.from_str("<R>"))
    pattern = np.arange(54, dtype=np.uint8)
    solver = BidirectionalSolver.from_actions_and_pattern(
        actions=actions,
        pattern=pattern,
        validator=None,
        optimize_indices=False,
    )
    permutations = [
        get_rubiks_cube_permutation(sequence=MoveSequence.from_str("R"), move_meta=move_meta),
        get_rubiks_cube_permutation(sequence=MoveSequence.from_str("R'"), move_meta=move_meta),
    ]

    summary = solver.search_many(
        permutations=permutations,
        max_solutions_per_permutation=1,
        min_search_depth=0,
        max_search_depth=1,
        max_time=10.0,
        side=SearchSide.normal,
    )

    assert summary.status is Status.Success
    assert len(summary.solutions) == 2
    by_root = {solution.permutation_index: str(solution.sequence) for solution in summary.solutions}
    assert by_root[0] == "R'"
    assert by_root[1] == "R"
