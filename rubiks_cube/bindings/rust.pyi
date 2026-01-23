import numpy as np

def add_one(x: int, n: int) -> int: ...
def bubble_sort(arr: np.ndarray) -> np.ndarray: ...
def bidirectional_solver(
    initial_permutation: np.ndarray,
    actions: list[np.ndarray],
    pattern: np.ndarray,
    min_search_depth: int,
    max_search_depth: int,
    n_solutions: int,
) -> list[list[int]] | None: ...
