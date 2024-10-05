from __future__ import annotations

from typing import Any

from rubiks_cube.move import format_string_to_generator
from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.move.sequence import cleanup


class MoveGenerator:
    """Rubiks cube move generator represented with a set of sequences."""

    def __init__(self, generator: str | set[MoveSequence] | None = None) -> None:
        """Initialize the move generator.

        Args:
            generator (str | set[MoveSequence] | None, optional):
                String of format "<Seq1, Seq2, ...>" or set of move sequences. Defaults to None.
        """
        if generator is None:
            self.generator = set()
        elif isinstance(generator, str):
            assert generator.startswith("<") and generator.endswith(
                ">"
            ), "Invalid move generator format!"
            sequence_list = format_string_to_generator(generator)
            self.generator = set([MoveSequence(seq) for seq in sequence_list])
        else:
            self.generator = generator

    def __str__(self) -> str:
        if not self.generator:
            return "<>"
        return "<" + ", ".join(sorted([str(seq) for seq in self.generator], key=len)) + ">"

    def __repr__(self) -> str:
        return f'MoveGenerator("{str(self)}")'

    def __len__(self) -> int:
        return len(self.generator)

    def __add__(self, other: MoveGenerator | set[MoveSequence]) -> MoveGenerator:
        if isinstance(other, MoveGenerator):
            return MoveGenerator(self.generator | other.generator)
        elif isinstance(other, set):
            return MoveGenerator(self.generator | other)

    def __radd__(self, other: MoveGenerator | set[MoveSequence]) -> MoveGenerator:
        if isinstance(other, MoveGenerator):
            return MoveGenerator(other.generator | self.generator)
        elif isinstance(other, set):
            return MoveGenerator(other | self.generator)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MoveGenerator):
            return self.generator == other.generator
        return False

    def __ne__(self, other: Any) -> bool:
        if isinstance(other, MoveGenerator):
            return self.generator != other.generator
        return True

    def __iter__(self) -> Any:
        for sequence in self.generator:
            yield sequence

    def __contains__(self, item: str) -> bool:
        return item in self.generator

    def __bool__(self) -> bool:
        return bool(self.generator)

    def __copy__(self) -> MoveGenerator:
        return MoveGenerator(generator=self.generator.copy())

    def __lt__(self, other: MoveGenerator | list[str]) -> bool:
        return len(self) < len(other)

    def __le__(self, other: MoveGenerator | list[str]) -> bool:
        return len(self) <= len(other)

    def __gt__(self, other: MoveGenerator | list[str]) -> bool:
        return len(self) > len(other)

    def __ge__(self, other: MoveGenerator | list[str]) -> bool:
        return len(self) >= len(other)


def cleanup_all(generator: MoveGenerator) -> MoveGenerator:
    """Cleanup all sequences in a move generator.

    Args:
        generator (MoveGenerator): Move generator.

    Returns:
        MoveGenerator: Cleaned move generator.
    """
    return MoveGenerator(set(cleanup(seq) for seq in generator))


def remove_empty(generator: MoveGenerator) -> MoveGenerator:
    """Remove empty sequences from a move generator.

    Args:
        generator (MoveGenerator): Move generator.

    Returns:
        MoveGenerator: Move generator without empty sequences.
    """
    return MoveGenerator(set(seq for seq in generator if seq))


def remove_inversed(generator: MoveGenerator) -> MoveGenerator:
    """Remove duplicate sequences that are inverses of each other.

    Args:
        generator (MoveGenerator): Move generator.

    Returns:
        MoveGenerator: Move generator without inverse duplicates.
    """
    new_generator = set()
    for seq in generator:
        clean_inv_seq = cleanup(~seq)
        if clean_inv_seq not in new_generator and seq not in new_generator:
            # Use the sequence with the shortest representation
            if len(str(seq)) <= len(str(clean_inv_seq)):
                new_generator.add(seq)
            else:
                new_generator.add(clean_inv_seq)

    return MoveGenerator(new_generator)


def simplify(generator: MoveGenerator) -> MoveGenerator:
    """Simplify a move generator by following these "rules":
    - Cleanup all sequences in the generator
    - Remove empty sequences
    - Remove sequences that are inverse of each other
    - (Remove sequences that are spanned by other sequences)

    Args:
        generator (MoveGenerator): Move generator.

    Returns:
        MoveGenerator: Simplified move generator.
    """
    generator = cleanup_all(generator)
    generator = remove_empty(generator)
    generator = remove_inversed(generator)

    return generator


def main() -> None:
    gen = MoveGenerator("<(R)R' (),(R'), R RR, R,xLw,R2'F, (R), ((R')R),, R'>")
    simple_gen = simplify(gen)
    control_gen = simplify(simple_gen)

    print("Initial generator:", gen)
    print("Simplyfied generator:", simple_gen)
    assert simple_gen == control_gen, "Simplify function failed!"


if __name__ == "__main__":
    main()
