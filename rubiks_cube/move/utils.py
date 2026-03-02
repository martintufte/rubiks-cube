from __future__ import annotations

import re

from rubiks_cube.configuration.regex import ROTATION_SEARCH
from rubiks_cube.group.so3 import canonicalize_sequence
from rubiks_cube.group.so3 import rotate_face


def is_rotation(move: str) -> bool:
    """Return True if the move is a rotation.

    Args:
        move (str): Move to check.

    Returns:
        bool: True if the move is a rotation.
    """
    return bool(re.search(ROTATION_SEARCH, move))


def invert_move(move: str) -> str:
    """Invert a move.

    Args:
        move (str): Move to invert.

    Returns:
        str: Inverted move.
    """
    if move.endswith("2"):
        return move
    return move[:-1] if move.endswith("'") else move + "'"


def rotate_move(move: str, rotation: str) -> str:
    """Apply a rotation by mapping the move to the new move.

    Args:
        move (str): Move to rotate.
        rotation (str): Rotation to apply.

    Returns:
        str: Rotated move.
    """
    assert is_rotation(rotation), f"Rotation {rotation} must be a rotation!"
    face = move[0]
    new_face = rotate_face(face, rotation)

    return move.replace(face, new_face)


def combine_rotations(rotations: list[str]) -> list[str]:
    """Collapse rotations in a sequence to a standard rotations.

    Args:
        rotations (list[str]): List of rotations.

    Returns:
        list[str]: List of canonical rotations.
    """
    return list(canonicalize_sequence(rotations))


def strip_move(move: str) -> str:
    """Strip a move of parenthesis.

    Args:
        move (str): Move to strip.

    Returns:
        str: Stripped move without NISS notation.
    """
    if move.startswith("("):
        move = move[1:]
    if move.endswith(")"):
        move = move[:-1]
    return move


def unstrip_move(move: str) -> str:
    """Decorate a move with parentheses.

    Args:
        move (str): Move to unstrip.

    Returns:
        str: Unstripped move with NISS notation.
    """
    if not move.startswith("("):
        move = "(" + move
    if not move.endswith(")"):
        move = move + ")"
    return move
