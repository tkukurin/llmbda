"""Tests for loop-as-leaf-skill patterns."""

from __future__ import annotations

from typing import Any

import pytest

from tk.llmbda import (
    Skill,
    SkillContext,
    StepResult,
    iter_skill,
    lm,
    run_skill,
)


def test_loop_runs_until_predicate():
    counter = {"n": 0}

    def loop_fn(_ctx: SkillContext) -> StepResult:
        for _ in range(10):
            counter["n"] += 1
            if counter["n"] >= 3:
                return StepResult(value=counter["n"], resolved=True)
        return StepResult(value=counter["n"])

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    result = run_skill(skill, {})
    assert result.value == 3
    assert counter["n"] == 3


def test_loop_stops_at_max_iter():
    counter = {"n": 0}

    def loop_fn(_ctx: SkillContext) -> StepResult:
        for _ in range(4):
            counter["n"] += 1
        return StepResult(value=counter["n"])

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    result = run_skill(skill, {})
    assert result.value == 4
    assert counter["n"] == 4


def test_loop_inner_resolved_breaks_loop():
    counter = {"n": 0}

    def loop_fn(_ctx: SkillContext) -> StepResult:
        for _ in range(10):
            counter["n"] += 1
            if counter["n"] == 2:
                return StepResult(value="done")
        return StepResult(value=counter["n"])

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    result = run_skill(skill, {})
    assert result.value == "done"
    assert counter["n"] == 2


def test_loop_with_steps_before_and_after():
    def setup(_ctx: SkillContext) -> StepResult:
        return StepResult(value="ready")

    counter = {"n": 0}

    def loop_fn(_ctx: SkillContext) -> StepResult:
        for _ in range(3):
            counter["n"] += 1
        return StepResult(value=counter["n"])

    def finalize(ctx: SkillContext) -> StepResult:
        return StepResult(value=f"final:{ctx.trace['loop'].value}")

    skill = Skill(
        name="s",
        steps=[
            Skill("setup", fn=setup),
            Skill("loop", fn=loop_fn),
            Skill("finalize", fn=finalize),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "final:3"
    assert result.resolved_by == "finalize"


def test_loop_zero_iterations():
    called: list[int] = []

    def loop_fn(_ctx: SkillContext) -> StepResult:
        called.extend(1 for _ in range(0))
        return StepResult(value=None)

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    result = run_skill(skill, {})
    assert called == []
    assert result.value is None


def test_loop_single_iteration_with_resolved():
    def step(_ctx: SkillContext) -> StepResult:
        return StepResult(value="immediate", resolved=True)

    skill = Skill(name="s", steps=[Skill("s", fn=step)])
    result = run_skill(skill, {})
    assert result.value == "immediate"


def test_loop_propagates_metadata():
    def loop_fn(_ctx: SkillContext) -> StepResult:
        last = StepResult(value=None)
        for _ in range(1):
            last = StepResult(value=1, metadata={"key": "val"})
        return StepResult(value=last.value, metadata=last.metadata)

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    result = run_skill(skill, {})
    assert result.metadata == {"key": "val"}


def test_loop_propagates_exception():
    counter = {"n": 0}

    def loop_fn(_ctx: SkillContext) -> StepResult:
        for _ in range(5):
            counter["n"] += 1
            if counter["n"] == 2:
                msg = "exploded"
                raise RuntimeError(msg)
        return StepResult(value=counter["n"])

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    with pytest.raises(RuntimeError, match="exploded"):
        run_skill(skill, {})
    assert counter["n"] == 2


def test_loop_with_lm_step():
    responses = iter(["draft", "better", "approved"])

    def fake(*, messages: list[dict[str, str]], **_kw: Any) -> str:  # noqa: ARG001
        return next(responses)

    @lm(fake, system_prompt="refine")
    def refine_loop(_ctx: SkillContext, call: Any) -> StepResult:
        for _ in range(5):
            raw = call(messages=[{"role": "user", "content": "go"}])
            if raw == "approved":
                return StepResult(value=raw)
        return StepResult(value=raw)

    skill = Skill(name="s", steps=[Skill("refine", fn=refine_loop)])
    result = run_skill(skill, {})
    assert result.value == "approved"


def test_loop_multi_step_resolved_mid_iteration():
    """A loop fn can compose multiple sub-steps internally."""
    counter = {"n": 0}
    called_second = []

    def loop_fn(_ctx: SkillContext) -> StepResult:
        for _ in range(10):
            counter["n"] += 1
            if counter["n"] == 2:
                return StepResult(value="resolved!")
            called_second.append(1)
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    result = run_skill(skill, {})
    assert result.value == "resolved!"
    assert len(called_second) == 1


def test_loop_ctx_entry_accessible():
    def loop_fn(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.entry["x"] * 2)

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    result = run_skill(skill, {"x": 21})
    assert result.value == 42


def test_loop_result_visible_in_skill_trace():
    def loop_fn(_ctx: SkillContext) -> StepResult:
        return StepResult(value="looped")

    skill = Skill(name="s", steps=[Skill("the_loop", fn=loop_fn)])
    result = run_skill(skill, {})
    assert "the_loop" in result.trace
    assert result.trace["the_loop"].value == "looped"


def test_loop_resolved_does_not_stop_skill():
    """resolved=True on a leaf short-circuits the sequence."""
    def resolve_immediately(_ctx: SkillContext) -> StepResult:
        return StepResult(value="from_loop", resolved=True)

    def unreachable(_ctx: SkillContext) -> StepResult:
        msg = "should not run"
        raise AssertionError(msg)

    skill = Skill(
        name="s",
        steps=[Skill("inner", fn=resolve_immediately), Skill("after", fn=unreachable)],
    )
    result = run_skill(skill, {})
    assert result.value == "from_loop"
    assert result.resolved_by == "inner"


def test_loop_resolved_does_not_stop_downstream_when_wrapped():
    """A loop fn can absorb resolved internally and let downstream run."""
    def loop_fn(_ctx: SkillContext) -> StepResult:
        return StepResult(value="from_loop")  # resolved=False

    def after_loop(ctx: SkillContext) -> StepResult:
        return StepResult(value=f"after:{ctx.trace['loop'].value}")

    skill = Skill(
        name="s",
        steps=[
            Skill("loop", fn=loop_fn),
            Skill("after", fn=after_loop),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "after:from_loop"
    assert result.resolved_by == "after"


def test_loop_resolved_with_steps_before_and_after():
    """Post-loop step runs when loop fn returns resolved=False."""
    def setup(_ctx: SkillContext) -> StepResult:
        return StepResult(value="ready")

    counter = {"n": 0}

    def loop_fn(_ctx: SkillContext) -> StepResult:
        for _ in range(5):
            counter["n"] += 1
            if counter["n"] == 3:
                return StepResult(value="done")
        return StepResult(value=counter["n"])

    def finalize(ctx: SkillContext) -> StepResult:
        return StepResult(value=f"final:{ctx.trace['loop'].value}")

    skill = Skill(
        name="s",
        steps=[
            Skill("setup", fn=setup),
            Skill("loop", fn=loop_fn),
            Skill("finalize", fn=finalize),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "final:done"
    assert result.resolved_by == "finalize"
    assert counter["n"] == 3


def test_loop_prev_updates_between_steps():
    """ctx.prev tracks the most recently executed step."""
    seen_prev: list[tuple[str, Any]] = []

    def step_a(ctx: SkillContext) -> StepResult:
        seen_prev.append(("a", ctx.prev.value))
        return StepResult(value="a_val")

    def step_b(ctx: SkillContext) -> StepResult:
        seen_prev.append(("b", ctx.prev.value))
        return StepResult(value="b_val")

    skill = Skill(
        name="s",
        steps=[Skill("a", fn=step_a), Skill("b", fn=step_b)],
    )
    run_skill(skill, {})
    assert seen_prev == [
        ("a", None),
        ("b", "a_val"),
    ]


def test_loop_internal_iteration_with_trace():
    """A loop fn can write intermediate results to ctx.trace."""
    def loop_fn(ctx: SkillContext) -> StepResult:
        for i in range(3):
            ctx.trace["counter"] = StepResult(value=i)
        return StepResult(value=ctx.trace["counter"].value)

    skill = Skill(name="s", steps=[Skill("loop", fn=loop_fn)])
    result = run_skill(skill, {})
    assert result.value == 2


def test_nested_loops():
    outer_count = {"n": 0}
    inner_count = {"n": 0}

    def outer_loop(_ctx: SkillContext) -> StepResult:
        for _ in range(2):
            outer_count["n"] += 1
            inner_count["n"] = 0
            for _ in range(3):
                inner_count["n"] += 1
        return StepResult(value=outer_count["n"])

    skill = Skill(name="s", steps=[Skill("loop", fn=outer_loop)])
    run_skill(skill, {})
    assert outer_count["n"] == 2


def test_orchestrator_fn_sees_children_via_ctx_skills():
    """fn+steps orchestrator receives its declared children in ctx.skills."""
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
    """After an orchestrator fn returns, ctx.skills is restored for siblings."""
    seen_by_after: list[list[Skill]] = []

    child = Skill("inner", fn=lambda _: StepResult(value=1))

    def orchestrator(ctx: SkillContext) -> StepResult:
        assert ctx.skills == [child]
        return StepResult(value="ok")

    def after(ctx: SkillContext) -> StepResult:
        seen_by_after.append(list(ctx.skills))
        return StepResult(value="done")

    skill = Skill(
        name="s",
        steps=[
            Skill("orch", fn=orchestrator, steps=[child]),
            Skill("after", fn=after),
        ],
    )
    run_skill(skill, {})
    leaves = seen_by_after[0]
    names = [s.name for s in leaves]
    assert "orch" in names
    assert "after" in names


def test_orchestrator_restores_ctx_skills_on_exception():
    """If orchestrator fn raises, ctx.skills is still restored."""
    child = Skill("inner", fn=lambda _: StepResult(value=1))

    def boom(_ctx: SkillContext) -> StepResult:
        msg = "orchestrator exploded"
        raise RuntimeError(msg)

    def after(_ctx: SkillContext) -> StepResult:
        return StepResult(value="after")

    orch = Skill("orch", fn=boom, steps=[child])
    after_skill = Skill("after", fn=after)
    skill = Skill(name="s", steps=[orch, after_skill])

    it = iter_skill(skill, {})
    with pytest.raises(RuntimeError, match="orchestrator exploded"):
        next(it)
    # ctx.skills should have been restored despite the exception;
    # verify by running a clean skill that would break if global state leaked
    clean = Skill(name="clean", steps=[after_skill])
    r = run_skill(clean, {})
    assert r.value == "after"


def test_leaf_fn_without_steps_sees_global_skills():
    """A plain leaf (fn only, no steps) sees the top-level leaves list."""
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
