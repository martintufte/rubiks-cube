from __future__ import annotations


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
