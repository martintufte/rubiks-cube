import logging
from typing import Final
from typing import Literal

import numpy as np

from rubiks_cube.configuration import CUBE_SIZE
from rubiks_cube.configuration.types import CubePattern
from rubiks_cube.configuration.types import CubePermutation
from rubiks_cube.state.pattern import get_empty_pattern
from rubiks_cube.state.permutation import create_permutations
from rubiks_cube.state.tracing import corner_trace
from rubiks_cube.tag.cubex import get_cubexes

LOGGER: Final = logging.getLogger(__name__)


def get_rubiks_cube_pattern(
    tag: str = "solved",
    subset: str | None = None,
    cube_size: int = CUBE_SIZE,
) -> CubePattern:
    """Get a matchable Rubik's cube pattern.

    Args:
        tag (str, optional): Tag to solve. Defaults to None.
        subset (str | None, optional): Subset of the tag. Defaults to None.
        permutation (CubePermutation | None, optional): Permutation of the cube. Defaults to None.
        cube_size (int, optional): Size of the cube. Defaults to CUBE_SIZE.
    """
    if tag == "none":
        return get_empty_pattern(cube_size=cube_size)

    cubexes = get_cubexes(sort_strategy="none", cube_size=cube_size)
    if tag not in cubexes:
        raise ValueError("Cannot create the pattern for the given tag and cube size.")

    cubex = cubexes[tag]
    if subset is None:
        idx = 0
    elif subset in cubex.names:
        idx = cubex.names.index(subset)
    else:
        raise ValueError("Subset does not exist in the given tag.")

    pattern = cubex.patterns[idx]

    return pattern


def autotag_permutation(
    permutation: CubePermutation,
    sort_strategy: Literal["entropy", "none"] = "none",
    default: str = "none",
    cube_size: int = CUBE_SIZE,
) -> str:
    """Tag the permutation.
    1. Find the tag corresponding to the state.
    2. Post-process the tag if necessary.

    Args:
        permutation (CubePermutation): Cube permutation.
        default_tag (str, optional): Dafualt tag. Defaults to "none".

    Returns:
        str: Tag of the state.
    """

    cubexes = get_cubexes(sort_strategy=sort_strategy, cube_size=cube_size)

    for tag, cbx in cubexes.items():
        if cbx.match(permutation):
            return_tag = tag
            break
    else:
        return_tag = default

    # TODO: This code works, but should be replaced w non-stochastic method!
    # If uses on average ~2 moves to differentiate between real/fake HTR
    # It recognizes if it is real/fake HTR by corner-tracing
    if return_tag == "htr-like":
        real_htr_traces = ["", "2c2c2c2c"]
        fake_htr_traces = [
            "3c2c2c",
            "2c2c2c",
            "4c3c",
            "4c",
            "2c",
            "3c2c",
            "4c2c2c",
            "3c",
        ]
        # real/fake = ['3c3c', '4c2c', '2c2c', '4c4c']

        rng = np.random.default_rng(seed=42)
        permutations = create_permutations()
        temp_state = np.copy(permutation)
        while return_tag == "htr-like":
            trace = corner_trace(temp_state)
            if trace in real_htr_traces:
                return_tag = "htr"
            elif trace in fake_htr_traces:
                return_tag = "fake-htr"
            else:
                move = rng.choice(["L2", "R2", "U2", "D2", "F2", "B2"], size=1)[0]
                temp_state = temp_state[permutations[move]]

    return return_tag


def autotag_step(
    initial_permutation: CubePermutation,
    final_permutation: CubePermutation,
    cube_size: int = CUBE_SIZE,
) -> str:
    """Tag the step.

    Args:
        initial_permutation (CubePermutation): Initial permutation.
        final_permutation (CubePermutation): Final permutation.

    Returns:
        str: Tag of the step.
    """

    if np.array_equal(initial_permutation, final_permutation):
        return "rotation"

    initial_tag = autotag_permutation(initial_permutation, cube_size=cube_size)
    final_tag = autotag_permutation(final_permutation, cube_size=cube_size)

    step_dict = {
        "eo -> eo": "drm",
        "eo -> dr": "dr",
        "dr -> htr": "htr",
        "dr-fb -> htr": "htr",
        "dr-lr -> htr": "htr",
        "dr-ud -> htr": "htr",
        "dr -> fake-htr": "fake-htr",
        "htr -> solved": "solved",
        "cross -> x-cross": "first-pair",
        "x-cross -> xx-cross": "second-pair",
        "x-cross -> xx-cross-adjacent": "second-pair",
        "xx-cross -> xxx-cross": "third-pair",
        "xx-cross-adjacent -> xxx-cross": "third-pair",
        "x-cross -> xxx-cross": "second-pair + third-pair",
        "xx-cross -> f2l": "last-pair",
        "xxx-cross -> f2l": "fourth-pair",
        "xxx-cross -> f2l-eo": "fourth-pair + eo",
        "xxx-cross -> f2l-ep-co": "fourth-pair + oll",
        "xxx-cross -> f2l-face": "fourth-pair + oll",
        "f2l -> f2l-face": "oll",
        "f2l -> solved": "ll",
        "f2l -> f2l-layer": "oll + pll",
        "f2l-face -> f2l-layer": "pll",
        "f2l-face -> solved": "pll",
        "f2l-eo -> f2l-face": "oll",
        "f2l-eo -> f2l-layer": "zbll",
        "f2l-eo -> solved": "zbll",
        "f2l-layer -> solved": "auf",
        "f2l-ep-co -> f2l-layer": "pll",
        "f2l-ep-co -> solved": "pll",
    }

    step = f"{initial_tag} -> {final_tag}"
    if initial_tag == "none" and final_tag != "none":
        return final_tag
    return step_dict.get(step, step)
