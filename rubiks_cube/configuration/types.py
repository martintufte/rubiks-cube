from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# Arrays representing Rubik's cube masks, patterns, and permutations
CubeMask: TypeAlias = npt.NDArray[np.bool_]
CubePattern: TypeAlias = npt.NDArray[np.uint]
CubePermutation: TypeAlias = npt.NDArray[np.uint]
CubeColor: TypeAlias = npt.NDArray[np.str_]
CubeState: TypeAlias = CubeMask | CubePattern | CubePermutation | CubeColor

# Normal arrays
BoolArray: TypeAlias = npt.NDArray[np.bool_]

# Range of cube sizes
CubeRange: TypeAlias = tuple[int | None, int | None]
