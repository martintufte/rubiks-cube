from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm

from rubiks_cube.bindings.rust import add_one as add_one_rust
from rubiks_cube.bindings.rust import bubble_sort as bubble_sort_rust


def add_one(x: int, n: int) -> int:
    for _ in range(n):
        x += 1
    return x


def bubble_sort_np(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError("bubble_sort_np only supports 1D arrays")

    # work on a copy so we don't mutate the caller's array
    result = arr.copy()
    n = result.size

    if n < 2:
        return result

    for i in range(n):
        swapped = False

        # last i elements are already in place
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True

        # early exit if already sorted
        if not swapped:
            break

    return result


def benchmark_add() -> None:
    n = 10_000_000
    a = 1

    # Python
    t0 = timer()
    _b = add_one(a, n)
    py_time = timer() - t0
    print(py_time)

    # Rust
    t0 = timer()
    _b2 = add_one_rust(a, n)
    rust_time = timer() - t0
    print(rust_time)

    print(f"Rust is {round(py_time / rust_time, 2)}x faster!")


def benchmark_bubble_sort() -> None:
    n = 100
    min_len = 100
    max_len = 1000
    rng = np.random.default_rng()

    # Prepare input
    arrs = []
    for _ in range(n):
        # Pick random length in bounds (inclusive)
        length = rng.integers(min_len, max_len + 1)

        # Create random u64 vector of that length
        arr = rng.integers(
            low=0,
            high=np.iinfo(np.uint64).max,
            size=length,
            dtype=np.uint64,
        )

        arrs.append(arr)

    # Python
    python_times = []
    rust_times = []

    for arr in tqdm(arrs):
        t0 = timer()
        _b = bubble_sort_np(arr)
        py_time = timer() - t0
        python_times.append(py_time)

        # Rust
        t0 = timer()
        _b2 = bubble_sort_rust(arr)
        rust_time = timer() - t0
        rust_times.append(rust_time)

    # Print speedup
    speedup = [
        py_time / rust_time for py_time, rust_time in zip(python_times, rust_times, strict=True)
    ]
    print(f"Rust is {round(sum(speedup) / n, 2)}x faster!")


if __name__ == "__main__":
    benchmark_bubble_sort()
