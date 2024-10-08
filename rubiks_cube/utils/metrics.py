import re

from rubiks_cube.configuration import METRIC
from rubiks_cube.configuration.enumeration import Metric


def count_length(moves: list[str], metric: Metric = METRIC) -> int:
    """Count the length of a sequence of moves.
    ETM: Execution Turn Metric
    HTM: Half Turn Metric
    STM: Slice Turn Metric
    QTM: Quarter Turn Metric

    Args:
        moves (list[str]): List of moves.
        metric (Metric, optional): Metric type. Defaults to METRIC.

    Returns:
        int: Length of the sequence.
    """

    count = len(moves) - sum(move.strip() == "" for move in moves)
    slices = sum(bool(re.search("[MES]", move)) for move in moves)
    rotations = sum(bool(re.search("[xyz]", move)) for move in moves)

    if metric is Metric.ETM:
        return count
    elif metric is Metric.HTM:
        return count + slices - rotations
    elif metric is Metric.STM:
        return count - rotations
    elif metric is Metric.QTM:
        d_count = sum(bool(re.search("[2]", move)) for move in moves)
        d_slices = sum(bool(re.search("[MES]2", move)) for move in moves)
        d_rotations = sum(bool(re.search("[xyz]2", move)) for move in moves)
        return count + slices - rotations + d_count + d_slices - d_rotations


def quarter_turn_parity(moves: list[str]) -> bool:
    """Find the quarter turn parity of a sequence of moves.

    Args:
        moves (list[str]): List of moves.

    Returns:
        bool: Parity of the sequence. True if even, False if odd.
    """
    return bool(count_length(moves, Metric.QTM) % 2)
