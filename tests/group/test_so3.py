from __future__ import annotations

from rubiks_cube.group.so3 import DEFAULT_STATE_ORDER
from rubiks_cube.group.so3 import IDENTITY_STATE
from rubiks_cube.group.so3 import apply_rotation_token
from rubiks_cube.group.so3 import compose_states
from rubiks_cube.group.so3 import get_cayley_table
from rubiks_cube.group.so3 import get_so3_elements
from rubiks_cube.group.so3 import state_from_sequence


def test_group_has_24_elements() -> None:
    elements = get_so3_elements()
    assert len(elements) == 24
    assert set(elements) == set(DEFAULT_STATE_ORDER)


def test_known_generator_states() -> None:
    assert state_from_sequence(["x"]) == "FD"
    assert state_from_sequence(["y"]) == "UR"
    assert state_from_sequence(["z"]) == "LF"


def test_apply_rotation_token_matches_expected_transition() -> None:
    assert apply_rotation_token("UF", "x") == "FD"
    assert apply_rotation_token("UF", "y") == "UR"
    assert apply_rotation_token("UF", "z") == "LF"


def test_cayley_table_identity_row_and_column() -> None:
    table = get_cayley_table()

    for state in DEFAULT_STATE_ORDER:
        assert table[IDENTITY_STATE][state] == state
        assert table[state][IDENTITY_STATE] == state


def test_group_inverse_exists_for_each_element() -> None:
    table = get_cayley_table()

    for left in DEFAULT_STATE_ORDER:
        inverses = [
            right
            for right in DEFAULT_STATE_ORDER
            if table[left][right] == IDENTITY_STATE and table[right][left] == IDENTITY_STATE
        ]
        assert len(inverses) == 1


def test_group_associativity() -> None:
    for a in DEFAULT_STATE_ORDER:
        for b in DEFAULT_STATE_ORDER:
            for c in DEFAULT_STATE_ORDER:
                left = compose_states(compose_states(a, b), c)
                right = compose_states(a, compose_states(b, c))
                assert left == right
