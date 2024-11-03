import re

from rubiks_cube.configuration import CUBE_SIZE
from rubiks_cube.configuration.types import CubePermutation
from rubiks_cube.move.algorithm import MoveAlgorithm
from rubiks_cube.move.generator import MoveGenerator
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.state import get_rubiks_cube_state


def get_action_space(
    generator: MoveGenerator | None = None,
    algorithms: list[MoveAlgorithm] | None = None,
    expand: bool = True,
    cube_size: int = CUBE_SIZE,
) -> dict[str, CubePermutation]:
    """Get the action space from the move generator and from the algorithms.

    Args:
        generator (MoveGenerator): Move generator.
        algorithms (list[MoveAlgorithm] | None): List of algorithms to include in the action space.
        expand (bool): Expand the actions to include all possible moves generated by the move.
        cube_size (int): Size of the cube.

    Returns:
        dict[str, CubePermutation]: Action space.
    """

    actions: dict[str, CubePermutation] = {}

    if generator is not None:
        for sequence in generator:
            if expand:
                for expanded_sequence in expanded_sequences(sequence):
                    actions[str(expanded_sequence)] = get_rubiks_cube_state(
                        sequence=expanded_sequence,
                        cube_size=cube_size,
                    )
            else:
                permutation = get_rubiks_cube_state(
                    sequence=sequence,
                    cube_size=cube_size,
                )
                actions[str(sequence)] = permutation

    if algorithms is not None:
        for algorithm in algorithms:
            assert algorithm.name not in actions, f"Algorithm {algorithm.name} already in actions!"
            assert (
                algorithm.cube_range[0] is None or algorithm.cube_range[0] <= cube_size
            ), f"Cube size {cube_size} is too small for algorithm {algorithm.name}!"
            assert (
                algorithm.cube_range[1] is None or algorithm.cube_range[1] >= cube_size
            ), f"Cube size {cube_size} is too large for algorithm {algorithm.name}!"
            actions[algorithm.name] = get_rubiks_cube_state(
                sequence=algorithm.sequence,
                cube_size=cube_size,
            )

    return actions


def expanded_sequences(sequence: MoveSequence) -> list[MoveSequence]:
    """This function expands a sequence into a list of sequences if the move is
    a sequence of length one and the move matches the standard pattern and the move
    is not a double move.

    Args:
        sequence (MoveSequence): The move sequence to expand.

    Returns:
        list[str]: List of expanded move sequences.
    """

    if len(sequence) != 1:
        return [sequence]

    move = sequence.moves[0]
    standard_pattern = re.compile(r"^([23456789]?[LRFBUD][w])(['2]?)$|^([LRFBUDxyzMES])(['2]?)$")
    match = standard_pattern.match(move)

    if match is not None:
        core = match.group(1) or match.group(3)
        modifier = match.group(2) or match.group(4)

        if modifier == "2":
            return [sequence]

        return [
            MoveSequence([f"{core}"]),
            MoveSequence([f"{core}'"]),
            MoveSequence([f"{core}2"]),
        ]

    return [sequence]
