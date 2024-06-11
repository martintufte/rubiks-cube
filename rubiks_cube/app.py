from typing import Any

import streamlit as st

from rubiks_cube.permutation import SOLVED_STATE
from rubiks_cube.permutation import get_permutation
from rubiks_cube.permutation.tracing import is_solved
from rubiks_cube.graphics.plotting import plot_cube_state
from rubiks_cube.tag.enumerations import Progress
from rubiks_cube.utils.formatter import format_string
from rubiks_cube.utils.formatter import is_valid_symbols
from rubiks_cube.utils.formatter import remove_comment
from rubiks_cube.utils.move import is_valid_moves
from rubiks_cube.utils.move import string_to_moves
from rubiks_cube.utils.sequence import Sequence
from rubiks_cube.utils.sequence import split_normal_inverse
from rubiks_cube.utils.sequence import unniss
from rubiks_cube.utils.sequence import cleanup


st.set_page_config(
    page_title="Fewest Moves Solver",
    page_icon="rubiks_cube/data/resources/favicon.png",
    layout="centered",
)


default_session: dict[str, Any] = {
    "scramble": Sequence(),
    "user": Sequence(),
    "tool": Sequence(),
    "premoves": True,
    "invert": False,
    "beta_features": False,
    "permutation": SOLVED_STATE,
}

for key, default in default_session.items():
    if key not in st.session_state:
        setattr(st.session_state, key, default)


def parse_user_input(user_input: str) -> list[str]:
    """
    Parse user input lines and return the move list:
    - Remove comments
    - Replace definitions provided by the user
    - Replace substitutions
    - Remove skip characters
    - Check for valid symbols
    - Check for valid moves
    - Combine all moves into a single list of moves
    """
    user_lines = []
    additional_chars = ""
    skip_chars = ","
    definition_symbol = "="
    sub_start = "["
    sub_end = "]"

    definitions = {}
    substitutions = []

    lines = user_input.strip().split("\n")
    n_lines = len(lines)

    for i, line_input in enumerate(reversed(lines)):
        line = remove_comment(line_input)
        if line.strip() == "":
            continue
        for definition, definition_moves in definitions.items():
            line = line.replace(definition, definition_moves)
        for char in skip_chars:
            line = line.replace(char, "")

        if definition_symbol in line:
            definition, definition_moves = line.split(definition_symbol)
            definition = definition.strip()
            definition_moves = definition_moves.strip()
            assert is_valid_moves(string_to_moves(definition_moves)), \
                "Definition moves are invalid!"
            if definition[0] == sub_start and definition[-1] == sub_end:
                substitutions.append(definition_moves)
                continue
            else:
                assert len(definition) == 1, \
                    "Definition must be a single character!"
                assert not is_valid_symbols(definition), \
                    "Definition must not be an inbuild symbol!"
                assert len(definition_moves) > 0, \
                    "Definition must have at least one move!"
                if not is_valid_symbols(definition_moves, additional_chars):
                    st.warning(f"Invalid symbols entered at line {n_lines-i}")
                    break
                definitions[definition] = definition_moves
                additional_chars += definition
                continue
        else:
            if sub_start in line and sub_end in line:
                start_idx = line.index(sub_start)
                end_idx = line.index(sub_end)
                to_replace = line[start_idx+1:end_idx]
                if not is_valid_moves(string_to_moves(to_replace)):
                    st.warning(f"Invalid rewrite at line {n_lines-i}")
                    break
                if substitutions:
                    line = line[:line.index(sub_start)] + \
                        substitutions.pop() + line[line.index(sub_end)+1:]
                else:
                    line = line.replace(sub_start, "").replace(sub_end, "")

        if not is_valid_symbols(line, additional_chars):
            st.warning(f"Invalid symbols entered at line {n_lines-i}")
            break
        line_moves_str = format_string(line)
        line_moves = string_to_moves(line_moves_str)
        if not is_valid_moves(line_moves):
            st.warning(f"Invalid moves entered at line {n_lines-i}")
            break
        else:
            user_lines.append(line_moves)

    user_moves = []
    for line in reversed(user_lines):
        user_moves += line

    return user_moves


def render_settings() -> None:
    """Render the settings bar."""
    col1, col2, col3, _ = st.columns(4)
    st.session_state.premoves = col1.toggle(
        label="Premoves",
        value=st.session_state.premoves,
    )
    st.session_state.invert = col2.toggle(
        label="Invert",
        value=st.session_state.invert,
    )
    st.session_state.beta_features = col3.toggle(
        label="Experimental",
        value=st.session_state.beta_features,
    )


def tag_progress(normal: Sequence, inverse: Sequence) -> tuple[str, Sequence]:
    """
    Tag the user progress as "solved" or "draft". Later: "Skeleton"
    Unnisses the sequence if it solves the cube.
    """
    user_permutation = get_permutation(
        sequence=(normal + ~inverse),
        ignore_rotations=True,
        from_permutation=st.session_state.permutation,
    )
    if is_solved(user_permutation):
        tag = Progress.solved
        out_sequence = cleanup(unniss(st.session_state.user))
    else:
        tag = Progress.draft
        out_sequence = st.session_state.user

    return tag.value, out_sequence


def render_experimental(sequence: Sequence) -> None:
    """Experimental features.

    Args:
        sequence (Sequence): Sequence to experiment on.
    """

    from rubiks_cube.tag import autotag_sequence

    st.write("Auto-tagging:")
    st.text(autotag_sequence(sequence))


def main() -> None:
    """Render the main page."""

    st.title("Fewest Moves Solver")
    scramble_input = st.text_input("Scramble", placeholder="R' U' F ...")
    scramble = remove_comment(scramble_input)

    if scramble.strip() == "":
        st.session_state.scramble = Sequence()
        st.info("Enter a scramble to get started!")
        return

    if not is_valid_symbols(scramble):
        st.error("Invalid symbols entered!")
        return

    scramble_moves_str = format_string(scramble)
    scramble_moves = string_to_moves(scramble_moves_str)
    if not is_valid_moves(scramble_moves):
        st.error("Invalid moves entered!")
        return

    st.session_state.scramble = Sequence(scramble_moves)
    st.session_state.permutation = get_permutation(
        st.session_state.scramble
    )

    fig = plot_cube_state(st.session_state.permutation)
    st.pyplot(fig, use_container_width=True)

    user_input = st.text_area("Moves", placeholder="Moves  // Comment\n...")

    if user_input.strip() == "":
        st.session_state.user = Sequence()
        st.info("Enter some moves to get started or use the tools!")
        return

    st.session_state.user = cleanup(Sequence(parse_user_input(user_input)))
    normal, inverse = split_normal_inverse(st.session_state.user)
    tag, progress = tag_progress(normal, inverse)

    out_string = f"{str(progress)}  // {tag} ({len(progress)})"
    st.text_input(label=tag, value=out_string, label_visibility="collapsed")

    render_settings()

    if st.session_state.premoves:
        full_sequence = ~inverse + st.session_state.scramble + normal
    else:
        full_sequence = st.session_state.scramble + st.session_state.user

    full_sequence = unniss(full_sequence)
    if st.session_state.invert:
        full_sequence = ~ full_sequence

    fig_user = plot_cube_state(get_permutation(full_sequence))
    st.pyplot(fig_user, use_container_width=True)

    # Experimental features
    if st.session_state.beta_features:
        render_experimental(sequence=full_sequence)


if __name__ == "__main__":
    main()