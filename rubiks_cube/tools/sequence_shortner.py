import streamlit as st

from rubiks_cube.state.permutation import SOLVED_STATE
from rubiks_cube.state.permutation import apply_moves
from rubiks_cube.state.permutation.tracing import count_similar
from rubiks_cube.state.permutation.tracing import count_solved
from rubiks_cube.state.permutation.tracing import is_solved
from rubiks_cube.move.sequence import MoveSequence


def generate_cube_states(init_perm, depth=3):
    all_states = {}

    return all_states


def execute_ss(command):
    """Execute a Sequence Shortner command."""

    return command


def render_ss_settings():
    """Render the settings bar for Sequence Shortner."""
    max_length = st.slider(
        label="Max length",
        min_value=3,
        max_value=10,
        value=3,
    )
    col1, col2, col3, _ = st.columns(4)
    look_for_rewrites = col1.toggle(
        label="Include rewrites",
        value=True,
    )
    recursive = col2.toggle(
        label="Recursive",
        value=True,
    )
    execute = col3.button(
        label="Execute"
    )
    return max_length, look_for_rewrites, recursive, execute


def render_sequence_shortner():
    """Sequence shortner tool."""

    # Settings
    max_length, look_for_rewrites, recursive, execute = render_ss_settings()

    if execute:
        with st.spinner("Looking for shortned sequence..."):
            moves = st.session_state.user.moves

            permutation = apply_moves(SOLVED_STATE, moves)

            output = execute_ss(
                f"shorten {max_length} {look_for_rewrites} {recursive} {moves}"
            )
            st.write(output)
            st.write(permutation)


class SequenceShortner():
    """ Sequence Shortner class. """
    def __init__(self):
        self.name = "Sequence shortner"
        pass

    def render(self):
        render_sequence_shortner()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


if __name__ == "__main__":

    p = apply_moves(SOLVED_STATE, MoveSequence("R U R' U' R U R' U' R U R' U'"))  # noqa
    q = apply_moves(SOLVED_STATE, MoveSequence("R' U2 R"))

    print("Solved:", is_solved(p))
    print("Number of solved pieces:", count_solved(p))
    print("Number of similar pieces:", count_similar(p, q))
