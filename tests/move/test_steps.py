import pytest

from rubiks_cube.move.sequence import MoveSequence
from rubiks_cube.move.steps import MoveSteps


class TestMoveStepsBasics:
    def test_empty_initialization(self) -> None:
        steps = MoveSteps()
        assert len(steps) == 0
        assert str(steps) == "None"
        assert not steps

    def test_initialization_from_move_sequences(self) -> None:
        steps = MoveSteps([MoveSequence.from_str("R U"), MoveSequence.from_str("R' U'")])
        assert len(steps) == 2
        assert steps[0] == MoveSequence.from_str("R U")
        assert steps[1] == MoveSequence.from_str("R' U'")

    def test_from_strings(self) -> None:
        steps = MoveSteps.from_strings(["R U", "R' U'"])
        assert steps == MoveSteps([MoveSequence.from_str("R U"), MoveSequence.from_str("R' U'")])

    def test_index_and_slice(self) -> None:
        steps = MoveSteps.from_strings(["R", "U", "F"])
        assert steps[0] == MoveSequence.from_str("R")
        sliced = steps[1:]
        assert list(sliced) == [MoveSequence.from_str("U"), MoveSequence.from_str("F")]

    def test_iteration(self) -> None:
        steps = MoveSteps.from_strings(["R", "U"])
        assert list(steps) == [MoveSequence.from_str("R"), MoveSequence.from_str("U")]

    def test_add_move_steps(self) -> None:
        left = MoveSteps.from_strings(["R U"])
        right = MoveSteps.from_strings(["R' U'"])
        result = left + right
        assert result == MoveSteps.from_strings(["R U", "R' U'"])

    def test_add_sequence_of_move_sequence(self) -> None:
        left = MoveSteps.from_strings(["R U"])
        right = [MoveSequence.from_str("F"), MoveSequence.from_str("F'")]
        result = left + right
        assert result == MoveSteps.from_strings(["R U", "F", "F'"])

    def test_to_sequence(self) -> None:
        steps = MoveSteps.from_strings(["R U", "R' U'"])
        assert steps.to_sequence() == MoveSequence.from_str("R U R' U'")

    def test_without_empty(self) -> None:
        steps = MoveSteps([MoveSequence.from_str("R"), MoveSequence(), MoveSequence.from_str("U")])
        assert steps.without_empty() == MoveSteps.from_strings(["R", "U"])

    def test_with_step_returns_new_instance(self) -> None:
        steps = MoveSteps.from_strings(["R"])
        updated = steps.with_step(MoveSequence.from_str("U"))
        assert steps == MoveSteps.from_strings(["R"])
        assert updated == MoveSteps.from_strings(["R", "U"])


def test_invalid_step_type_raises() -> None:
    with pytest.raises(TypeError):
        MoveSteps(steps=["R"])  # type: ignore[list-item]


def test_apply_local_update_not_implemented() -> None:
    steps = MoveSteps.from_strings(["R U"])
    with pytest.raises(NotImplementedError):
        steps.apply_local_update(0, "R U R'")


def test_resolve_subsets_not_implemented() -> None:
    steps = MoveSteps.from_strings(["R U"])
    with pytest.raises(NotImplementedError):
        steps.resolve_subsets()
