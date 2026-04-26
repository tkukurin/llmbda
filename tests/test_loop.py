"""Tests for the loop primitive."""

from __future__ import annotations

from typing import Any

import pytest

from tk.llmbda import (
    Skill,
    Step,
    StepContext,
    StepResult,
    lm,
    loop,
    run_skill,
)


def test_loop_runs_until_predicate():
    counter = {"n": 0}

    def increment(ctx: StepContext) -> StepResult:
        counter["n"] += 1
        return StepResult(value=counter["n"], resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(
                Step("inc", increment),
                name="loop",
                max_iter=10,
                until=lambda ctx: ctx.prior["inc"].value >= 3,
            ),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == 3
    assert counter["n"] == 3


def test_loop_stops_at_max_iter():
    counter = {"n": 0}

    def increment(ctx: StepContext) -> StepResult:
        counter["n"] += 1
        return StepResult(value=counter["n"], resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(
                Step("inc", increment),
                name="loop",
                max_iter=4,
            ),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == 4
    assert counter["n"] == 4


def test_loop_inner_resolved_breaks_loop():
    counter = {"n": 0}

    def resolve_on_two(ctx: StepContext) -> StepResult:
        counter["n"] += 1
        if counter["n"] == 2:
            return StepResult(value="done")
        return StepResult(value=counter["n"], resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(
                Step("check", resolve_on_two),
                name="loop",
                max_iter=10,
            ),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "done"
    assert counter["n"] == 2


def test_loop_prior_updates_each_iteration():
    """Inner steps see updated prior from previous iterations."""
    def writer(ctx: StepContext) -> StepResult:
        prev = ctx.prior.get("writer")
        n = (prev.value + 1) if prev else 0
        return StepResult(value=n, resolved=False)

    def reader(ctx: StepContext) -> StepResult:
        return StepResult(value=f"saw:{ctx.prior['writer'].value}", resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(
                Step("writer", writer),
                Step("reader", reader),
                name="loop",
                max_iter=3,
            ),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "saw:2"


def test_loop_with_steps_before_and_after():
    def setup(ctx: StepContext) -> StepResult:
        return StepResult(value="ready", resolved=False)

    counter = {"n": 0}

    def bump(ctx: StepContext) -> StepResult:
        counter["n"] += 1
        return StepResult(value=counter["n"], resolved=False)

    def finalize(ctx: StepContext) -> StepResult:
        return StepResult(value=f"final:{ctx.prior['loop'].value}")

    skill = Skill(
        name="s",
        steps=[
            Step("setup", setup),
            loop(
                Step("bump", bump),
                name="loop",
                max_iter=3,
            ),
            Step("finalize", finalize),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "final:3"
    assert result.resolved_by == "finalize"


def test_loop_zero_max_iter():
    called = []

    def step(ctx: StepContext) -> StepResult:
        called.append(1)
        return StepResult(value="x", resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(Step("s", step), name="loop", max_iter=0),
        ],
    )
    result = run_skill(skill, {})
    assert called == []
    assert result.value is None


def test_loop_single_iteration_with_until():
    def step(ctx: StepContext) -> StepResult:
        return StepResult(value="immediate", resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(
                Step("s", step),
                name="loop",
                max_iter=10,
                until=lambda ctx: True,
            ),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "immediate"


def test_loop_propagates_metadata():
    def step(ctx: StepContext) -> StepResult:
        return StepResult(value=1, metadata={"key": "val"})

    skill = Skill(
        name="s",
        steps=[loop(Step("s", step), name="loop", max_iter=1)],
    )
    result = run_skill(skill, {})
    assert result.metadata == {"key": "val"}


def test_loop_propagates_exception():
    counter = {"n": 0}

    def boom(ctx: StepContext) -> StepResult:
        counter["n"] += 1
        if counter["n"] == 2:
            raise RuntimeError("exploded")
        return StepResult(value=counter["n"], resolved=False)

    skill = Skill(
        name="s",
        steps=[loop(Step("boom", boom), name="loop", max_iter=5)],
    )
    with pytest.raises(RuntimeError, match="exploded"):
        run_skill(skill, {})
    assert counter["n"] == 2


def test_loop_with_lm_step():
    responses = iter(["draft", "better", "approved"])

    def fake(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        return next(responses)

    @lm(fake, system_prompt="refine")
    def refine(ctx: StepContext, call: Any) -> StepResult:
        raw = call(messages=[{"role": "user", "content": "go"}])
        return StepResult(value=raw, resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(
                Step("refine", refine),
                name="loop",
                max_iter=5,
                until=lambda ctx: ctx.prior["refine"].value == "approved",
            ),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "approved"


def test_loop_multiple_steps_resolved_mid_iteration():
    """If the first step in a multi-step loop resolves, second step is skipped."""
    counter = {"n": 0}

    def maybe_resolve(ctx: StepContext) -> StepResult:
        counter["n"] += 1
        if counter["n"] == 2:
            return StepResult(value="resolved!")
        return StepResult(value=counter["n"], resolved=False)

    called_second = []

    def second(ctx: StepContext) -> StepResult:
        called_second.append(1)
        return StepResult(value="second", resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(
                Step("first", maybe_resolve),
                Step("second", second),
                name="loop",
                max_iter=10,
            ),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == "resolved!"
    assert len(called_second) == 1  # only ran in iteration 1


def test_loop_ctx_entry_accessible():
    def step(ctx: StepContext) -> StepResult:
        return StepResult(value=ctx.entry["x"] * 2)

    skill = Skill(
        name="s",
        steps=[loop(Step("s", step), name="loop", max_iter=1)],
    )
    result = run_skill(skill, {"x": 21})
    assert result.value == 42


def test_loop_result_visible_in_skill_trace():
    def step(ctx: StepContext) -> StepResult:
        return StepResult(value="looped", resolved=False)

    skill = Skill(
        name="s",
        steps=[loop(Step("s", step), name="the_loop", max_iter=2)],
    )
    result = run_skill(skill, {})
    assert "the_loop" in result.trace
    assert result.trace["the_loop"].value == "looped"


def test_nested_loops():
    outer_count = {"n": 0}
    inner_count = {"n": 0}

    def outer_step(ctx: StepContext) -> StepResult:
        outer_count["n"] += 1
        inner_count["n"] = 0  # reset inner for each outer iteration
        return StepResult(value=outer_count["n"], resolved=False)

    def inner_step(ctx: StepContext) -> StepResult:
        inner_count["n"] += 1
        return StepResult(value=inner_count["n"], resolved=False)

    skill = Skill(
        name="s",
        steps=[
            loop(
                Step("outer", outer_step),
                loop(
                    Step("inner", inner_step),
                    name="inner_loop",
                    max_iter=3,
                ),
                name="outer_loop",
                max_iter=2,
            ),
        ],
    )
    result = run_skill(skill, {})
    assert outer_count["n"] == 2