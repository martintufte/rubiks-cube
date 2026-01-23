from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Final

from rubiks_cube.autotagger import get_rubiks_cube_pattern
from rubiks_cube.bindings.rust import bidirectional_solver
from rubiks_cube.configuration.enumeration import Goal
from rubiks_cube.move.generator import MoveGenerator
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.representation import get_rubiks_cube_state
from rubiks_cube.solver.actions import get_actions

LOGGER: Final = logging.getLogger(__name__)


def main() -> None:
    """Test calling the bidirectional solver."""
    goal = Goal.solved
    subset = None
    cube_size = 3
    sequence = MoveSequence("M2 U M U2 M' U M2")
    generator = MoveGenerator("<M, U>")

    if generator is None:
        generator = MoveGenerator(generator="<L, R, F, B, U, D>")

    LOGGER.info(f"Solving with {goal=} and {subset=}.")

    # Setup solver
    actions = get_actions(generator=generator, algorithms=None, cube_size=cube_size)
    pattern = get_rubiks_cube_pattern(goal=goal, subset=subset, cube_size=cube_size)

    initial_permutation = get_rubiks_cube_state(
        sequence=sequence,
        initial_permutation=None,
        invert_after=False,
        cube_size=cube_size,
    )

    min_search_depth = 0
    max_search_depth = 10
    n_solutions = 5

    # Solve the permutation with the class
    start_time = timer()
    _solutions = bidirectional_solver(
        initial_permutation,
        list(actions.values()),  # Enumerated 0..n_actions
        pattern,
        min_search_depth,
        max_search_depth,
        n_solutions,
    )
    walltime = timer() - start_time

    LOGGER.info(f"Walltime: {walltime}")


if __name__ == "__main__":
    main()
