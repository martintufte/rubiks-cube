from __future__ import annotations

from typing import Final

from rubiks_cube.beam_search.interface import BeamPlan
from rubiks_cube.beam_search.interface import BeamStep
from rubiks_cube.beam_search.interface import Transition
from rubiks_cube.configuration.enumeration import Goal
from rubiks_cube.configuration.enumeration import Symmetry
from rubiks_cube.move.generator import MoveGenerator

DR_PLAN: Final[BeamPlan] = BeamPlan(
    name="dr",
    cube_size=3,
    steps=[
        BeamStep(
            goal=Goal.eo,
            variations=[Symmetry.lr, Symmetry.fb, Symmetry.ud],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.none: MoveGenerator.from_str("<L, R, F, B, U, D>"),
                },
            ),
            max_search_depth=6,
            max_solutions=30,
        ),
        BeamStep(
            goal=Goal.dr,
            variations=[Symmetry.lr, Symmetry.fb, Symmetry.ud],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.lr: MoveGenerator.from_str("<L2, R2, F, B, U, D>"),
                    Symmetry.fb: MoveGenerator.from_str("<L, R, F2, B2, U, D>"),
                    Symmetry.ud: MoveGenerator.from_str("<L, R, F, B, U2, D2>"),
                },
                check_contained=True,
                expand_variations=True,
            ),
            max_search_depth=10,
            max_solutions=10,
        ),
    ],
)

HTR_PLAN: Final[BeamPlan] = BeamPlan(
    name="htr",
    cube_size=3,
    steps=[
        BeamStep(
            goal=Goal.eo,
            variations=[Symmetry.lr, Symmetry.fb, Symmetry.ud],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.none: MoveGenerator.from_str("<L, R, F, B, U, D>"),
                },
            ),
            max_search_depth=6,
            max_solutions=30,
        ),
        BeamStep(
            goal=Goal.dr,
            variations=[Symmetry.lr, Symmetry.fb, Symmetry.ud],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.lr: MoveGenerator.from_str("<L2, R2, F, B, U, D>"),
                    Symmetry.fb: MoveGenerator.from_str("<L, R, F2, B2, U, D>"),
                    Symmetry.ud: MoveGenerator.from_str("<L, R, F, B, U2, D2>"),
                },
                check_contained=True,
                expand_variations=True,
            ),
            max_search_depth=10,
            max_solutions=10,
        ),
        BeamStep(
            goal=Goal.htr,
            variations=[Symmetry.none],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.lr: MoveGenerator.from_str("<L, R, F2, B2, U2, D2>"),
                    Symmetry.ud: MoveGenerator.from_str("<L2, R2, F2, B2, U, D>"),
                    Symmetry.fb: MoveGenerator.from_str("<L2, R2, F, B, U2, D2>"),
                },
                expand_variations=True,
            ),
            max_search_depth=12,
            max_solutions=10,
        ),
    ],
)

SOLVED_PLAN: Final[BeamPlan] = BeamPlan(
    name="solved",
    cube_size=3,
    steps=[
        BeamStep(
            goal=Goal.eo,
            variations=[Symmetry.lr, Symmetry.fb, Symmetry.ud],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.none: MoveGenerator.from_str("<L, R, F, B, U, D>"),
                },
            ),
            max_search_depth=6,
            max_solutions=30,
        ),
        BeamStep(
            goal=Goal.dr,
            variations=[Symmetry.lr, Symmetry.fb, Symmetry.ud],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.lr: MoveGenerator.from_str("<L2, R2, F, B, U, D>"),
                    Symmetry.fb: MoveGenerator.from_str("<L, R, F2, B2, U, D>"),
                    Symmetry.ud: MoveGenerator.from_str("<L, R, F, B, U2, D2>"),
                },
                check_contained=True,
                expand_variations=True,
            ),
            max_search_depth=10,
            max_solutions=10,
        ),
        BeamStep(
            goal=Goal.htr,
            variations=[Symmetry.none],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.lr: MoveGenerator.from_str("<L, R, F2, B2, U2, D2>"),
                    Symmetry.fb: MoveGenerator.from_str("<L2, R2, F, B, U2, D2>"),
                    Symmetry.ud: MoveGenerator.from_str("<L2, R2, F2, B2, U, D>"),
                },
                expand_variations=True,
            ),
            max_search_depth=12,
            max_solutions=10,
        ),
        BeamStep(
            goal=Goal.solved,
            variations=[Symmetry.none],
            transition=Transition(
                search_side="prev",
                generator_map={
                    Symmetry.none: MoveGenerator.from_str("<L2, R2, F2, B2, U2, D2>"),
                },
                expand_variations=True,
            ),
            max_search_depth=12,
            max_solutions=5,
        ),
    ],
)


LEAVE_SLICE_PLAN: Final[BeamPlan] = BeamPlan(
    name="leave slice",
    cube_size=3,
    steps=[
        BeamStep(
            goal=Goal.eo,
            variations=[Symmetry.lr, Symmetry.fb, Symmetry.ud],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.none: MoveGenerator.from_str("<L, R, F, B, U, D>"),
                },
            ),
            max_search_depth=6,
            max_solutions=30,
        ),
        BeamStep(
            goal=Goal.dr,
            variations=[Symmetry.lr, Symmetry.fb, Symmetry.ud],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.lr: MoveGenerator.from_str("<L2, R2, F, B, U, D>"),
                    Symmetry.fb: MoveGenerator.from_str("<L, R, F2, B2, U, D>"),
                    Symmetry.ud: MoveGenerator.from_str("<L, R, F, B, U2, D2>"),
                },
                check_contained=True,
                expand_variations=True,
            ),
            max_search_depth=10,
            max_solutions=10,
        ),
        BeamStep(
            goal=Goal.htr,
            variations=[Symmetry.none],
            transition=Transition(
                search_side="both",
                generator_map={
                    Symmetry.lr: MoveGenerator.from_str("<L, R, F2, B2, U2, D2>"),
                    Symmetry.fb: MoveGenerator.from_str("<L2, R2, F, B, U2, D2>"),
                    Symmetry.ud: MoveGenerator.from_str("<L2, R2, F2, B2, U, D>"),
                },
                expand_variations=True,
            ),
            max_search_depth=12,
            max_solutions=10,
        ),
        BeamStep(
            goal=Goal.solved,
            variations=[Symmetry.none],
            transition=Transition(
                search_side="prev",
                generator_map={
                    Symmetry.none: MoveGenerator.from_str("<L2, R2, F2, B2, U2, D2>"),
                },
                expand_variations=True,
            ),
            max_search_depth=12,
            max_solutions=5,
        ),
        BeamStep(
            goal=Goal.leave_slice,
            variations=[Symmetry.m, Symmetry.s, Symmetry.e],
            transition=Transition(
                search_side="prev",
                generator_map={
                    Symmetry.lr: MoveGenerator.from_str("<L2, R2, F2, B2, U2, D2>"),
                    Symmetry.fb: MoveGenerator.from_str("<L2, R2, F2, B2, U2, D2>"),
                    Symmetry.ud: MoveGenerator.from_str("<L2, R2, F2, B2, U2, D2>"),
                },
                allowed_goals_by_prev_goal={
                    Symmetry.lr: frozenset({Symmetry.m}),
                    Symmetry.fb: frozenset({Symmetry.s}),
                    Symmetry.ud: frozenset({Symmetry.e}),
                },
                prev_goal_index=-2,
                expand_variations=True,
            ),
            max_search_depth=10,
            max_solutions=10,
        ),
    ],
)

BEAM_PLANS: Final[dict[str, BeamPlan]] = {
    SOLVED_PLAN.name: SOLVED_PLAN,
    LEAVE_SLICE_PLAN.name: LEAVE_SLICE_PLAN,
    HTR_PLAN.name: HTR_PLAN,
    DR_PLAN.name: DR_PLAN,
}
