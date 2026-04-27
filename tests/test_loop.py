"""Tests for loop-as-leaf and orchestrator behavior."""

from __future__ import annotations

from typing import Any

import pytest

from tk.llmbda import Skill, SkillContext, StepResult, iter_skill, run_skill


def test_loop_leaf_result_visible_in_trace():
    def loop_fn(_ctx: SkillContext) -> StepResult:
        value = None
        for i in range(3):
            value = i
        return StepResult(value=value)

    def after(ctx: SkillContext) -> StepResult:
        return StepResult(value=f"after:{ctx.trace['loop'].value}")

    skill = Skill(
        name="s",
        steps=[
            Skill("loop", fn=loop_fn),
            Skill("after", fn=after),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "after:2"
    assert result.trace["loop"].value == 2


def test_loop_leaf_can_absorb_internal_resolution():
    def loop_fn(_ctx: SkillContext) -> StepResult:
        for i in range(3):
            if i == 1:
                return StepResult(value="done")
        return StepResult(value="miss")

    def after(ctx: SkillContext) -> StepResult:
        return StepResult(value=f"after:{ctx.trace['loop'].value}")

    skill = Skill(
        name="s",
        steps=[
            Skill("loop", fn=loop_fn),
            Skill("after", fn=after),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "after:done"
    assert result.resolved_by == "after"


def test_loop_leaf_resolved_short_circuits_sequence():
    def loop_fn(_ctx: SkillContext) -> StepResult:
        return StepResult(value="done", resolved=True)

    def unreachable(_ctx: SkillContext) -> StepResult:
        msg = "should not run"
        raise AssertionError(msg)

    skill = Skill(
        name="s",
        steps=[
            Skill("loop", fn=loop_fn),
            Skill("after", fn=unreachable),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "done"
    assert result.resolved_by == "loop"


def test_loop_leaf_gets_ctx_entry_and_prev():
    seen_prev: list[Any] = []

    def setup(_ctx: SkillContext) -> StepResult:
        return StepResult(value=10)

    def loop_fn(ctx: SkillContext) -> StepResult:
        seen_prev.append(ctx.prev.value)
        total = ctx.prev.value
        for item in ctx.entry["items"]:
            total += item
        return StepResult(value=total)

    skill = Skill(
        name="s",
        steps=[
            Skill("setup", fn=setup),
            Skill("loop", fn=loop_fn),
        ],
    )
    result = run_skill(skill, {"items": [1, 2, 3]})
    assert result.value == 16
    assert seen_prev == [10]


def test_orchestrator_fn_sees_children_via_ctx_skills():
    seen_skills: list[list[Skill]] = []

    child_a = Skill("a", fn=lambda _: StepResult(value=1))
    child_b = Skill("b", fn=lambda _: StepResult(value=2))

    def orchestrator(ctx: SkillContext) -> StepResult:
        seen_skills.append(list(ctx.skills))
        return StepResult(value="ok")

    skill = Skill(name="orch", fn=orchestrator, steps=[child_a, child_b])
    run_skill(Skill(name="s", steps=[skill]), {})
    assert seen_skills == [[child_a, child_b]]


def test_orchestrator_restores_ctx_skills_on_success():
    seen_by_after: list[list[str]] = []

    child = Skill("inner", fn=lambda _: StepResult(value=1))

    def orchestrator(ctx: SkillContext) -> StepResult:
        assert ctx.skills == [child]
        return StepResult(value="ok")

    def after(ctx: SkillContext) -> StepResult:
        seen_by_after.append([s.name for s in ctx.skills])
        return StepResult(value="done")

    skill = Skill(
        name="s",
        steps=[
            Skill("orch", fn=orchestrator, steps=[child]),
            Skill("after", fn=after),
        ],
    )
    run_skill(skill, {})
    assert seen_by_after == [["orch", "after"]]


def test_orchestrator_restores_ctx_skills_on_exception():
    child = Skill("inner", fn=lambda _: StepResult(value=1))

    def boom(_ctx: SkillContext) -> StepResult:
        msg = "orchestrator exploded"
        raise RuntimeError(msg)

    def after(_ctx: SkillContext) -> StepResult:
        return StepResult(value="after")

    skill = Skill(
        name="s",
        steps=[
            Skill("orch", fn=boom, steps=[child]),
            Skill("after", fn=after),
        ],
    )

    it = iter_skill(skill, {})
    with pytest.raises(RuntimeError, match="orchestrator exploded"):
        next(it)

    clean = Skill(name="clean", steps=[Skill("after", fn=after)])
    assert run_skill(clean, {}).value == "after"


def test_leaf_fn_without_steps_sees_global_skills():
    seen_skills: list[list[str]] = []

    def leaf(ctx: SkillContext) -> StepResult:
        seen_skills.append([s.name for s in ctx.skills])
        return StepResult(value="ok")

    skill = Skill(
        name="s",
        steps=[
            Skill("a", fn=lambda _: StepResult(value=1)),
            Skill("b", fn=leaf),
        ],
    )
    run_skill(skill, {})
    assert seen_skills == [["a", "b"]]
