from __future__ import annotations


def strip_move(move: str) -> str:
    """Strip a decorated move.

    Args:
        move (str): Move to strip.

    Returns:
        str: Stripped move.
    """
    if move.startswith("("):
        move = move[1:]
    if move.endswith(")"):
        move = move[:-1]
    return move


def decorate_move(move: str, niss: bool = False) -> str:
    """Decorate a move with parentheses.

    Args:
        move (str): Move to unstrip.
        niss (bool): True if niss is active.

    Returns:
        str: Unstripped move.
    """
    if niss:
        move = "(" + move + ")"
    return move
