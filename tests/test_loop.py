"""Tests for orchestrator behavior."""

from __future__ import annotations

import pytest

from tk.llmbda import Skill, SkillContext, StepResult, fst_match, iter_skill, last, run_skill


def test_fst_match_returns_first_non_none():
    skill = Skill(
        name="s",
        steps=[
            Skill(
                "pick",
                fn=fst_match,
                steps=[
                    Skill("a", fn=lambda _: StepResult()),
                    Skill("b", fn=lambda _: StepResult(value="found")),
                    Skill("c", fn=lambda _: StepResult(value="late")),
                ],
            )
        ],
    )
    trace = run_skill(skill, {})
    assert last(trace).value == "found"


def test_fst_match_returns_empty_when_all_none():
    skill = Skill(
        name="s",
        steps=[
            Skill(
                "pick",
                fn=fst_match,
                steps=[
                    Skill("a", fn=lambda _: StepResult()),
                    Skill("b", fn=lambda _: StepResult()),
                ],
            )
        ],
    )
    trace = run_skill(skill, {})
    assert last(trace).value is None


def test_orchestrator_fn_receives_children_as_steps_arg():
    seen_steps: list[list[Skill]] = []

    child_a = Skill("a", fn=lambda _: StepResult(value=1))
    child_b = Skill("b", fn=lambda _: StepResult(value=2))

    def orchestrator(_ctx: SkillContext, steps: list[Skill]) -> StepResult:
        seen_steps.append(list(steps))
        return StepResult(value="ok")

    skill = Skill(name="orch", fn=orchestrator, steps=[child_a, child_b])
    run_skill(Skill(name="s", steps=[skill]), {})
    assert seen_steps == [[child_a, child_b]]


def test_orchestrator_can_run_children_via_run_skill():
    def orchestrator(ctx: SkillContext, steps: list[Skill]) -> StepResult:
        inner = Skill(name="inner", steps=steps)
        r = run_skill(inner, ctx.entry)
        return StepResult(value=last(r).value, meta={"ran": list(r)})

    skill = Skill(
        name="orch",
        fn=orchestrator,
        steps=[
            Skill("a", fn=lambda _: StepResult(value=1)),
            Skill("b", fn=lambda ctx: StepResult(value=ctx.prev.value + 1)),
        ],
    )
    trace = run_skill(Skill(name="s", steps=[skill]), {})
    assert last(trace).value == 2
    assert trace["orch"].meta["ran"] == ["a", "b"]


def test_orchestrator_early_exit_pattern():
    """Orchestrator stops on first non-None child result."""

    def first_match(ctx: SkillContext, steps: list[Skill]) -> StepResult:
        for s in steps:
            r = run_skill(Skill(name="_try", steps=[s]), ctx.entry)
            v = last(r)
            if v.value is not None:
                return v
        return StepResult()

    def try_cache(ctx: SkillContext) -> StepResult:
        cache = {"known": "cached-value"}
        return StepResult(value=cache.get(ctx.entry.get("key")))

    def compute(_ctx: SkillContext) -> StepResult:
        return StepResult(value="computed")

    skill = Skill(
        name="s",
        steps=[
            Skill(
                "lookup",
                fn=first_match,
                steps=[Skill("cache", fn=try_cache), Skill("compute", fn=compute)],
            ),
        ],
    )
    hit = run_skill(skill, {"key": "known"})
    assert last(hit).value == "cached-value"

    miss = run_skill(skill, {"key": "other"})
    assert last(miss).value == "computed"


def test_orchestrator_shares_outer_trace():
    def setup(_ctx: SkillContext) -> StepResult:
        return StepResult(value="setup_value")

    def orchestrator(ctx: SkillContext, _steps: list[Skill]) -> StepResult:
        return StepResult(value=f"saw:{ctx.trace['setup'].value}")

    skill = Skill(
        name="s",
        steps=[
            Skill("setup", fn=setup),
            Skill(
                "orch",
                fn=orchestrator,
                steps=[Skill("child", fn=lambda _: StepResult(value=1))],
            ),
        ],
    )
    trace = run_skill(skill, {})
    assert last(trace).value == "saw:setup_value"


def test_orchestrator_children_fresh_context():
    """Children run in a fresh SkillContext via nested run_skill."""

    def setup(_ctx: SkillContext) -> StepResult:
        return StepResult(value="outer")

    def orchestrator(ctx: SkillContext, steps: list[Skill]) -> StepResult:
        inner = run_skill(Skill(name="_", steps=steps), ctx.entry)
        return last(inner)

    def child(ctx: SkillContext) -> StepResult:
        assert ctx.trace.get("setup") is None
        return StepResult(value="child_ok")

    skill = Skill(
        name="s",
        steps=[
            Skill("setup", fn=setup),
            Skill("orch", fn=orchestrator, steps=[Skill("child", fn=child)]),
        ],
    )
    trace = run_skill(skill, {})
    assert last(trace).value == "child_ok"


def test_orchestrator_exception_does_not_corrupt_context():
    child = Skill("inner", fn=lambda _: StepResult(value=1))

    def boom(_ctx: SkillContext, _steps: list[Skill]) -> StepResult:
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
    assert last(run_skill(clean, {})).value == "after"


def test_orchestrator_retry_pattern():
    call_count = 0

    def flaky(_ctx: SkillContext) -> StepResult:
        nonlocal call_count
        call_count += 1
        valid = call_count >= 2
        return StepResult(value=f"attempt-{call_count}", meta={"valid": valid})

    def retry(ctx: SkillContext, steps: list[Skill]) -> StepResult:
        for _ in range(3):
            r = run_skill(Skill(name="_", steps=steps), ctx.entry)
            v = last(r)
            if v.meta.get("valid"):
                return v
        return v

    skill = Skill(
        name="s", steps=[Skill("retry", fn=retry, steps=[Skill("f", fn=flaky)])]
    )
    trace = run_skill(skill, {})
    assert last(trace).value == "attempt-2"
    assert last(trace).meta["valid"] is True
