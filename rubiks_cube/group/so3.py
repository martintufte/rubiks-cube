from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final
from typing import Sequence

FACE_ORDER: Final[tuple[str, ...]] = ("U", "D", "F", "B", "L", "R")

# Basic quarter turns around the fixed cube axes.
_X_FACE_MAP: Final[dict[str, str]] = {
    "U": "F",
    "F": "D",
    "D": "B",
    "B": "U",
    "L": "L",
    "R": "R",
}
_Y_FACE_MAP: Final[dict[str, str]] = {
    "U": "U",
    "D": "D",
    "F": "R",
    "R": "B",
    "B": "L",
    "L": "F",
}
_Z_FACE_MAP: Final[dict[str, str]] = {
    "U": "L",
    "L": "D",
    "D": "R",
    "R": "U",
    "F": "F",
    "B": "B",
}


def _compose_face_maps(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Compose two face maps where `left` is applied before `right`."""
    return {face: right[left[face]] for face in FACE_ORDER}


def _pow_face_map(face_map: dict[str, str], exponent: int) -> dict[str, str]:
    if exponent == 0:
        return {face: face for face in FACE_ORDER}
    result = {face: face for face in FACE_ORDER}
    for _ in range(exponent):
        result = _compose_face_maps(result, face_map)
    return result


ROTATION_FACE_MAPS: Final[dict[str, dict[str, str]]] = {
    "x": _X_FACE_MAP,
    "x'": _pow_face_map(_X_FACE_MAP, 3),
    "x2": _pow_face_map(_X_FACE_MAP, 2),
    "y": _Y_FACE_MAP,
    "y'": _pow_face_map(_Y_FACE_MAP, 3),
    "y2": _pow_face_map(_Y_FACE_MAP, 2),
    "z": _Z_FACE_MAP,
    "z'": _pow_face_map(_Z_FACE_MAP, 3),
    "z2": _pow_face_map(_Z_FACE_MAP, 2),
}

# State naming: XY means original X face points Up and original Y face points Front.
CANONICAL_SEQUENCES: Final[dict[str, tuple[str, ...]]] = {
    "UF": (),
    "UL": ("y'",),
    "UB": ("y2",),
    "UR": ("y",),
    "FU": ("x", "y2"),
    "FL": ("x", "y'"),
    "FD": ("x",),
    "FR": ("x", "y"),
    "RU": ("z'", "y'"),
    "RF": ("z'",),
    "RD": ("z'", "y"),
    "RB": ("z'", "y2"),
    "BU": ("x'",),
    "BL": ("x'", "y'"),
    "BD": ("x'", "y2"),
    "BR": ("x'", "y"),
    "LU": ("z", "y"),
    "LF": ("z",),
    "LD": ("z", "y'"),
    "LB": ("z", "y2"),
    "DF": ("z2",),
    "DL": ("x2", "y'"),
    "DB": ("x2",),
    "DR": ("x2", "y"),
}

IDENTITY_STATE: Final[str] = "UF"
DEFAULT_STATE_ORDER: Final[tuple[str, ...]] = tuple(CANONICAL_SEQUENCES.keys())


@dataclass(frozen=True, slots=True)
class SO3Element:
    state: str
    sequence: tuple[str, ...]
    face_map: tuple[str, ...]

    def as_face_map(self) -> dict[str, str]:
        return dict(zip(FACE_ORDER, self.face_map, strict=True))


def _face_map_key(face_map: dict[str, str]) -> tuple[str, ...]:
    return tuple(face_map[face] for face in FACE_ORDER)


@lru_cache(maxsize=1)
def _canonical_face_maps() -> dict[str, dict[str, str]]:
    """Face-map realization of each canonical state sequence.

    The fold order intentionally follows the legacy orientation semantics:
    new_map = compose(token_map, current_map).
    """
    identity_map = {face: face for face in FACE_ORDER}
    by_state: dict[str, dict[str, str]] = {}

    for state, sequence in CANONICAL_SEQUENCES.items():
        current_map = identity_map
        for token in sequence:
            if token not in ROTATION_FACE_MAPS:
                raise KeyError(f"Unknown rotation token in canonical sequence: {token}")
            current_map = _compose_face_maps(ROTATION_FACE_MAPS[token], current_map)
        by_state[state] = current_map

    if len({_face_map_key(face_map) for face_map in by_state.values()}) != 24:
        raise ValueError("Canonical SO(3) mappings are not unique.")

    return by_state


@lru_cache(maxsize=1)
def _state_by_face_map() -> dict[tuple[str, ...], str]:
    return {_face_map_key(face_map): state for state, face_map in _canonical_face_maps().items()}


def _state_from_face_map(face_map: dict[str, str]) -> str:
    key = _face_map_key(face_map)
    try:
        return _state_by_face_map()[key]
    except KeyError as exception:
        raise ValueError(f"Invalid orientation state from face map: {key}") from exception


@lru_cache(maxsize=1)
def get_so3_elements() -> dict[str, SO3Element]:
    """Build all 24 orientation-preserving cube rotations."""
    canonical_maps = _canonical_face_maps()
    by_state: dict[str, SO3Element] = {}

    for state in DEFAULT_STATE_ORDER:
        if state not in CANONICAL_SEQUENCES:
            raise ValueError(f"Missing canonical sequence for state: {state}")

        sequence = get_canonical_sequence(state)
        if state_from_sequence(sequence) != state:
            raise ValueError(f"Invalid canonical sequence for state: {state}")

        face_map = canonical_maps[state]
        by_state[state] = SO3Element(
            state=state,
            sequence=sequence,
            face_map=_face_map_key(face_map),
        )

    if len(by_state) != 24:
        raise ValueError(f"Expected 24 rotations, found {len(by_state)}")

    missing = set(by_state) ^ set(DEFAULT_STATE_ORDER)
    if missing:
        raise ValueError(f"State mismatch while building SO(3): {sorted(missing)}")

    return by_state


def apply_rotation_token(state: str, token: str) -> str:
    """Apply a single rotation token to an orientation state."""
    if token not in ROTATION_FACE_MAPS:
        raise KeyError(f"Unknown rotation token: {token}")
    if state not in CANONICAL_SEQUENCES:
        raise KeyError(f"Unknown state: {state}")

    current_map = _canonical_face_maps()[state]
    next_map = _compose_face_maps(ROTATION_FACE_MAPS[token], current_map)
    return _state_from_face_map(next_map)


def state_from_sequence(sequence: Sequence[str]) -> str:
    """Return the state after applying a token sequence from identity."""
    state = IDENTITY_STATE
    for token in sequence:
        state = apply_rotation_token(state, token)
    return state


def get_canonical_sequence(state: str) -> tuple[str, ...]:
    """Return the canonical move sequence for an orientation state."""
    if state not in CANONICAL_SEQUENCES:
        raise KeyError(f"Unknown state: {state}")
    return CANONICAL_SEQUENCES[state]


def canonicalize_sequence(sequence: Sequence[str]) -> tuple[str, ...]:
    """Collapse a sequence to the canonical representative of the same element."""
    return get_canonical_sequence(state_from_sequence(sequence))


def rotate_face(face: str, token: str) -> str:
    """Rotate a single face label by a rotation token."""
    if token not in ROTATION_FACE_MAPS:
        raise KeyError(f"Unknown rotation token: {token}")
    return ROTATION_FACE_MAPS[token].get(face, face)


def compose_states(left: str, right: str) -> str:
    """Compose two group elements as: first `left`, then `right`."""
    if left not in CANONICAL_SEQUENCES:
        raise KeyError(f"Unknown left state: {left}")
    if right not in CANONICAL_SEQUENCES:
        raise KeyError(f"Unknown right state: {right}")

    left_map = _canonical_face_maps()[left]
    right_map = _canonical_face_maps()[right]
    product_map = _compose_face_maps(left_map, right_map)
    return _state_from_face_map(product_map)


def get_cayley_table(order: tuple[str, ...] = DEFAULT_STATE_ORDER) -> dict[str, dict[str, str]]:
    """Return the Cayley table for the cube rotation group."""
    elements = get_so3_elements()
    for state in order:
        if state not in elements:
            raise KeyError(f"Unknown state in table order: {state}")

    return {left: {right: compose_states(left, right) for right in order} for left in order}
