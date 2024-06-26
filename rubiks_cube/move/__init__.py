import re

from rubiks_cube.utils.formatter import format_string


def format_string_to_moves(string: str) -> list[str]:
    """Format a string into a list of moves."""
    formatted_string = format_string(string)

    moves = []
    niss = False
    for move in formatted_string.split():
        stripped_move = strip_move(move)
        if move.startswith("("):
            niss = not niss
        moves.append("(" + stripped_move + ")" if niss else stripped_move)
        if move.endswith(")"):
            niss = not niss

    if is_valid_moves(moves):
        return moves
    raise ValueError(f"Invalid moves entered! got {moves}")


def is_valid_moves(moves: list[str]) -> bool:
    """Check if a list of moves uses valid Rubik's Cube notation."""

    pattern = r"^[I]?$|^[RLFBUD][w][2']?$|^[RLUDFBxyzMES][2']?$"
    return all(re.match(pattern, strip_move(move)) for move in moves)


def invert_move(move: str) -> str:
    """Invert a move."""
    if move.startswith("("):
        return "(" + invert_move(move[1:-1]) + ")"
    if move.endswith("'"):
        return move[:-1]
    elif move.endswith("2"):
        return move
    return move + "'"


def strip_move(move: str) -> str:
    """Strip a move of parentheses."""
    return move.replace("(", "").replace(")", "")


def is_rotation(move: str) -> bool:
    """Return True if the move is a rotation."""

    return bool(re.search('[ixyz]', move))


def rotate_move(move: str, rotation: str) -> str:
    """Apply a rotation by mapping the move to the new move."""
    assert is_rotation(rotation), f"Rotation {rotation} must be a rotation!"
    rotation_moves_dict = {
        "i": {},
        "x": {"F": "D", "D": "B", "B": "U", "U": "F"},
        "x'": {"F": "U", "U": "B", "B": "D", "D": "F"},
        "x2": {"F": "B", "U": "D", "B": "F", "D": "U"},
        "y": {"F": "R", "L": "F", "B": "L", "R": "B"},
        "y'": {"F": "L", "L": "B", "B": "R", "R": "F"},
        "y2": {"F": "B", "L": "R", "B": "F", "R": "L"},
        "z": {"U": "L", "R": "U", "D": "R", "L": "D"},
        "z'": {"U": "R", "R": "D", "D": "L", "L": "U"},
        "z2": {"U": "D", "R": "L", "D": "U", "L": "R"},
    }
    face = move[0]
    new_face = rotation_moves_dict[rotation].get(face, face)

    return move.replace(face, new_face)


def get_axis(move: str) -> str | None:
    """Get the axis of a move."""
    if move[0] in "FB":
        return "z"
    elif move[0] in "RL":
        return "x"
    elif move[0] in "UD":
        return "y"
    return None


def format_string_to_generator(gen_string: str) -> list[list[str]]:
    """Format a string into a set of moves."""

    gen_string = gen_string.strip()
    assert gen_string.startswith("<") and gen_string.endswith(">"), \
        "Invalid move generator format!"
    string_moves = gen_string[1:-1].split(",")

    generator = []
    for string in string_moves:
        moves = format_string_to_moves(string)
        if is_valid_moves(moves):
            generator.append(moves)
        else:
            raise ValueError(f"Invalid moves for generator! got {moves}")

    return generator


def main() -> None:
    string = "(fx R2) ()U2M(\t)' (L' Dw2 F2) b y'F'"
    moves = format_string_to_moves(string)
    print("Raw:", string)
    print("Formatted:", " ".join(moves).replace(") (", " "))


if __name__ == "__main__":
    main()
