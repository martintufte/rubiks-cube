import pytest

from rubiks_cube.formatting.parsing import parse_steps
from rubiks_cube.move.steps import MoveSteps


def test_parse_steps_returns_move_steps() -> None:
    parsed = parse_steps("R U\nF2")

    assert isinstance(parsed, MoveSteps)
    assert parsed == MoveSteps.from_strings(["R U", "F2"])


def test_parse_steps_empty_input() -> None:
    parsed = parse_steps("")
    assert parsed == MoveSteps()


def test_parse_steps_rejects_skeleton_mode() -> None:
    with pytest.raises(
        ValueError,
        match=r"Definitions, substitutions, and skeleton syntax are not supported at line 1\.",
    ):
        parse_steps("-> R U R' U'")
