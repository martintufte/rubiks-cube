import logging
from typing import Final

import numpy as np

from rubiks_cube.configuration.types import CubePattern
from rubiks_cube.configuration.types import CubePermutation
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.move.sequence import cleanup
from rubiks_cube.representation.utils import invert

LOGGER: Final = logging.getLogger(__name__)


# TODO: Optimizations for the bidirectional solver
# Action space / solver optimazations:
# - Find the terminal actions and use them for the first branching
# - Use the last moves to determine the next moves
# - Make use of action groupings to reduce the effective branching factor
# - Remove identity actions and combine equivalent actions
# Bidirectional search optimizations:
# - Search a given depth (burn = n) from one side before initial switch
# - Give a message to the user if no solution is reachable with infinite depth
# - Return state="solved" with no solutions if the cube is already solved
# - Investigate and implementing simple pruning techniques
# - Deep action space pruning to reduce branching factor further


def encode(permutation: CubePermutation, pattern: CubePattern) -> str:
    """Encode a permutation into a string using a pattern.

    Args:
        permutation (CubePermutation): Cube state.
        pattern (CubePattern): Pattern.

    Returns:
        str: Encoded string.
    """
    return str(pattern[permutation])


def bidirectional_solver(
    initial_permutation: CubePermutation,
    actions: dict[str, CubePermutation],
    pattern: CubePattern,
    max_search_depth: int = 10,
    n_solutions: int = 1,
) -> list[str] | None:
    """Bidirectional solver for the Rubik's cube.

    It uses a breadth-first search from both states to find the shortest path
    between two states and returns the optimal solution.

    Args:
        initial_permutation (CubePermutation): The initial permutation.
        actions (dict[str, CubePermutation]): A dictionary of actions and permutations.
        pattern (CubePattern): The pattern that must match.
        max_search_depth (int, optional): The maximum depth. Defaults to 10.
        n_solutions (int, optional): The number of solutions to find. Defaults to 1.

    Returns:
        list[str]: List of solutions.
    """
    initial_str = encode(initial_permutation, pattern)
    last_states_normal: dict[str, tuple[CubePattern, list[str]]] = {
        initial_str: (initial_permutation, [])
    }
    searched_states_normal: dict[str, tuple[CubePattern, list[str]]] = {
        initial_str: (initial_permutation, [])
    }

    # Last searched permutations and all searched states on inverse permutation
    identity = np.arange(initial_permutation.size)
    solved_str = encode(identity, pattern)
    last_states_inverse: dict[str, tuple[CubePattern, list[str]]] = {solved_str: (identity, [])}
    searched_states_inverse: dict[str, tuple[CubePattern, list[str]]] = {solved_str: (identity, [])}

    # Store the solutions as cleaned sequence for keys and unclear for values
    solutions: dict[str, str] = {}

    # Check if the initial state is solved
    LOGGER.info("Searching for solutions..")
    LOGGER.info("Search depth: 0")
    if initial_str in searched_states_inverse:
        LOGGER.info("Found solution")
        return []

    i = 0
    while i < max_search_depth:
        # Expand last searched states on normal permutation
        i += 1
        LOGGER.info(f"Search depth: {i}")
        new_searched_states_normal: dict[str, tuple[CubePattern, list[str]]] = {}
        for permutation, move_list in last_states_normal.values():
            for move, action in actions.items():
                new_permutation = permutation[action]
                new_state_str = encode(new_permutation, pattern)
                if new_state_str not in searched_states_normal:
                    new_move_list = [*move_list, move]
                    new_searched_states_normal[new_state_str] = (
                        new_permutation,
                        new_move_list,
                    )

                    # Check if inverse permutation is searched
                    new_inverse_str = encode(new_permutation, pattern)
                    if new_inverse_str in last_states_inverse:
                        solution = MoveSequence(new_move_list) + ~MoveSequence(
                            last_states_inverse[new_inverse_str][1]
                        )
                        solution_cleaned = str(cleanup(solution))
                        if solution_cleaned not in solutions:
                            solutions[solution_cleaned] = str(solution)
                            LOGGER.info(f"Found solution ({len(solutions)}/{n_solutions})")
                        if len(solutions) == n_solutions:
                            return list(solutions.values())

        searched_states_normal.update(new_searched_states_normal)
        last_states_normal = new_searched_states_normal

        # Expand last searched states on inverse permutation
        if i == max_search_depth:
            break
        i += 1
        LOGGER.info(f"Search depth: {i}")
        new_searched_states_inverse: dict[str, tuple[CubePattern, list[str]]] = {}
        for permutation, move_list in last_states_inverse.values():
            for move, action in actions.items():
                new_permutation = permutation[action]
                new_state_str = encode(new_permutation, pattern)
                if new_state_str not in searched_states_inverse:
                    new_move_list = [*move_list, move]
                    new_searched_states_inverse[new_state_str] = (
                        new_permutation,
                        new_move_list,
                    )

                    # Check if inverse permutation is searched
                    new_inverse_str = encode(invert(new_permutation), pattern)
                    if new_inverse_str in last_states_normal:
                        solution = MoveSequence(
                            last_states_normal[new_inverse_str][1] + new_move_list
                        )
                        solution_cleaned = str(cleanup(solution))
                        if solution_cleaned not in solutions:
                            solutions[solution_cleaned] = str(solution)
                            LOGGER.info(f"Found solution ({len(solutions)}/{n_solutions})")
                        if len(solutions) == n_solutions:
                            return list(solutions.values())

        searched_states_inverse.update(new_searched_states_inverse)
        last_states_inverse = new_searched_states_inverse

    if len(solutions) == 0:
        LOGGER.warning("No solution found")
        return None
    return list(solutions.values())
