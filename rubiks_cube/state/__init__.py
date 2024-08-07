import numpy as np

from rubiks_cube.state.permutation import get_solved_state
from rubiks_cube.state.permutation import create_permutations
from rubiks_cube.state.permutation.utils import invert
from rubiks_cube.move import is_rotation
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.move.sequence import cleanup
from rubiks_cube.move.sequence import decompose


def get_rubiks_cube_state(
    sequence: MoveSequence,
    initial_state: np.ndarray | None = None,
    orientate_after: bool = False,
    use_inverse: bool = True,
    invert_state: bool = False,
    cube_size: int = 3,
) -> np.ndarray:
    """Get the cube state from a sequence of moves.

    Args:
        sequence (MoveSequence): Rubiks cube move sequence.
        initial_state (np.ndarray, optional): Initial state of the cube.
            Defaults to SOLVED_STATE.
        orientate_after (bool, optional): Orientate to same orientation as the
            initial state. Defaults to False.
        use_inverse (bool, optional): Use the inverse part. Defaults to True.
        invert_state (bool, optional): Whether to invert. Defaults to False.

    Returns:
        np.ndarray: The Rubiks cube state.
    """

    if initial_state is None:
        initial_state = get_solved_state(size=cube_size)

    # Decompose the sequence
    normal_sequence, inverse_sequence = decompose(sequence)
    permutation_dict = create_permutations(size=cube_size)
    state = initial_state.copy()

    # Apply moves on inverse
    if inverse_sequence and use_inverse:
        inverse_state = get_rubiks_cube_state(
            sequence=inverse_sequence,
            initial_state=invert(state),
            orientate_after=orientate_after,
            invert_state=False,
            cube_size=cube_size,
        )
        state = invert(inverse_state)

    # Apply moves on normal
    for move in cleanup(normal_sequence):
        if orientate_after and is_rotation(move):
            break
        state = state[permutation_dict[move]]

    # Invert the sequence and initial state if 'invert_state'
    if invert_state:
        return invert(state)
    return state
