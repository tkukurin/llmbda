"""Smoke tests for tk.llmbda core."""

from __future__ import annotations

from typing import Any

import pytest

from tk.llmbda import (
    Skill,
    SkillResult,
    Step,
    StepContext,
    StepResult,
    iter_skill,
    run_skill,
)


def _noop(**_kw: Any) -> None:
    return None


def echo_step(ctx: StepContext) -> StepResult:
    """Step that returns the entry as-is."""
    return StepResult(value=ctx.entry)


def counting_step(ctx: StepContext) -> StepResult:
    """Step that counts prior steps and falls through."""
    return StepResult(
        value=len(ctx.prior),
        metadata={"seen": list(ctx.prior.keys())},
        resolved=False,
    )


class DummyResponse:
    def __init__(self, content):
        self.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": content})()})()
        ]


def test_single_step():
    skill = Skill(name="echo", steps=[Step("echo", echo_step)])
    result = run_skill(skill, {"x": 1}, _noop)
    assert isinstance(result, SkillResult)
    assert result.value == {"x": 1}
    assert result.skill == "echo"
    assert result.resolved_by == "echo"
    assert list(result.trace) == ["echo"]
    assert result.trace["echo"].value == {"x": 1}


def test_resolved_short_circuits():
    def _resolver(_ctx):
        return StepResult(value="stopped")

    def _unreachable(_ctx):
        msg = "should not be called"
        raise AssertionError(msg)

    skill = Skill(
        name="short",
        steps=[Step("resolver", _resolver), Step("unreachable", _unreachable)],
    )
    result = run_skill(skill, {}, _noop)
    assert result.value == "stopped"
    assert result.resolved_by == "resolver"
    assert list(result.trace) == ["resolver"]


def test_implicit_resolved_on_last_step():
    skill = Skill(
        name="chain",
        steps=[
            Step("a", counting_step),
            Step("b", counting_step),
            Step("c", counting_step),
        ],
    )
    result = run_skill(skill, {}, _noop)
    assert result.value == 2  # step c sees a and b
    assert result.resolved_by == "c"
    assert list(result.trace) == ["a", "b", "c"]


def test_prior_accumulates():
    def deposit(_ctx):
        return StepResult(value="first", metadata={"order": 1}, resolved=False)

    def check(ctx):
        prior_val = ctx.prior["deposit"].value
        return StepResult(value=f"saw {prior_val}")

    skill = Skill(
        name="acc",
        steps=[Step("deposit", deposit), Step("check", check)],
    )
    result = run_skill(skill, {}, _noop)
    assert result.value == "saw first"
    assert result.trace["deposit"].value == "first"
    assert result.trace["deposit"].metadata == {"order": 1}


def test_duplicate_step_names_raise_before_running():
    called: list[str] = []

    def _step(_ctx):
        called.append("ran")
        return StepResult(value=None, resolved=False)

    skill = Skill(
        name="dup",
        steps=[Step("same", _step), Step("same", _step)],
    )
    with pytest.raises(ValueError, match="same"):
        list(iter_skill(skill, {}, _noop))
    assert called == []


def test_empty_skill():
    skill = Skill(name="noop")
    result = run_skill(skill, {"x": 1}, _noop)
    assert result.value is None
    assert result.skill == "noop"
    assert result.resolved_by == "(empty)"
    assert result.trace == {}


def test_step_metadata_preserved_unchanged():
    def _with_meta(_ctx):
        return StepResult(value=42, metadata={"custom": "data", "extra": True})

    skill = Skill(name="meta", steps=[Step("with_meta", _with_meta)])
    result = run_skill(skill, {}, _noop)
    assert result.metadata == {"custom": "data", "extra": True}
    assert result.skill == "meta"
    assert result.resolved_by == "with_meta"


def test_step_metadata_not_mutated_by_runtime():
    """Runtime attrs live on SkillResult; step's metadata dict is untouched."""
    emitted = StepResult(value=1, metadata={"skill": "hijacked"})

    def _clashing(_ctx):
        return emitted

    skill = Skill(name="real", steps=[Step("clash", _clashing)])
    result = run_skill(skill, {}, _noop)
    assert result.skill == "real"
    assert result.resolved_by == "clash"
    assert emitted.metadata == {"skill": "hijacked"}  # unchanged
    assert result.metadata == {"skill": "hijacked"}


def test_iter_yields_each_step_in_order():
    skill = Skill(
        name="chain",
        steps=[
            Step("a", counting_step),
            Step("b", counting_step),
            Step("c", counting_step),
        ],
    )
    yielded = list(iter_skill(skill, {}, _noop))
    assert [name for name, _ in yielded] == ["a", "b", "c"]
    assert [r.value for _, r in yielded] == [0, 1, 2]


def test_iter_stops_after_resolved():
    def _stop(_ctx):
        return StepResult(value="done")

    def _never(_ctx):
        msg = "should not run"
        raise AssertionError(msg)

    skill = Skill(
        name="s",
        steps=[Step("stop", _stop), Step("never", _never)],
    )
    yielded = list(iter_skill(skill, {}, _noop))
    assert [name for name, _ in yielded] == ["stop"]


def test_iter_stop_decision_survives_result_mutation():
    called: list[str] = []

    def _stop(_ctx):
        return StepResult(value="done")

    def _never(_ctx):
        called.append("never")
        return StepResult(value="unreachable")

    skill = Skill(
        name="s",
        steps=[Step("stop", _stop), Step("never", _never)],
    )
    it = iter_skill(skill, {}, _noop)
    name, result = next(it)
    assert name == "stop"

    result.resolved = False

    with pytest.raises(StopIteration):
        next(it)
    assert called == []


def test_iter_break_early():
    seen: list[str] = []
    skill = Skill(
        name="chain",
        steps=[
            Step("a", counting_step),
            Step("b", counting_step),
            Step("c", counting_step),
        ],
    )
    for name, _ in iter_skill(skill, {}, _noop):
        seen.append(name)
        if name == "a":
            break
    assert seen == ["a"]


def test_iter_collects_into_dict():
    skill = Skill(
        name="chain",
        steps=[Step("a", counting_step), Step("b", counting_step)],
    )
    trace = dict(iter_skill(skill, {}, _noop))
    assert set(trace) == {"a", "b"}
    assert trace["b"].value == 1


def test_iter_empty_skill_yields_nothing():
    assert list(iter_skill(Skill(name="noop"), {}, _noop)) == []


def test_caller_prepends_step_system_prompt():
    """Runtime binds ctx.caller to auto-prepend the step's system_prompt."""
    seen: list[list[dict[str, str]]] = []

    def spy_caller(**kwargs):
        seen.append(kwargs["messages"])
        return "ok"

    def llm_step(ctx):
        ctx.caller(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value="done")

    skill = Skill(
        name="s",
        steps=[Step("llm", llm_step, system_prompt="be terse")],
    )
    run_skill(skill, {}, spy_caller)
    assert seen == [[
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]]


def test_caller_does_not_inject_prompt_without_messages():
    """System prompts are only added to explicit messages kwargs."""
    captured: list[dict[str, object]] = []

    def spy_caller(**kwargs):
        captured.append(kwargs)
        return "ok"

    def llm_step(ctx):
        ctx.caller(model="noop")
        return StepResult(value="done")

    skill = Skill(
        name="s",
        steps=[Step("llm", llm_step, system_prompt="be terse")],
    )
    run_skill(skill, {}, spy_caller)
    assert captured == [{"model": "noop"}]


def test_caller_passes_through_when_prompt_empty():
    """No system_prompt on the step => caller forwards messages unchanged."""
    seen: list[list[dict[str, str]]] = []

    def spy_caller(**kwargs):
        seen.append(kwargs["messages"])
        return "ok"

    def llm_step(ctx):
        ctx.caller(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Step("llm", llm_step)])
    run_skill(skill, {}, spy_caller)
    assert seen == [[{"role": "user", "content": "hi"}]]


def test_caller_rebinds_between_steps():
    """Each step sees a caller bound to its own prompt, not a prior step's."""
    captured: list[list[dict[str, str]]] = []

    def capturing(**kwargs):
        captured.append(kwargs["messages"])
        return "ok"

    def fn_a(ctx):
        ctx.caller(messages=[{"role": "user", "content": "a"}])
        return StepResult(value=None, resolved=False)

    def fn_b(ctx):
        ctx.caller(messages=[{"role": "user", "content": "b"}])
        return StepResult(value="done")

    skill = Skill(
        name="s",
        steps=[
            Step("a", fn_a, system_prompt="prompt-a"),
            Step("b", fn_b, system_prompt="prompt-b"),
        ],
    )
    run_skill(skill, {}, capturing)
    assert captured == [
        [{"role": "system", "content": "prompt-a"},
         {"role": "user", "content": "a"}],
        [{"role": "system", "content": "prompt-b"},
         {"role": "user", "content": "b"}],
    ]


def test_caller_available_in_context():
    recorded = []

    def spy_step(ctx):
        recorded.append(ctx.caller)
        return StepResult(value="ok")

    def fake_caller(**_kw) -> Any:
        return DummyResponse('{"x": 1}')

    skill = Skill(name="spy", steps=[Step("spy", spy_step)])
    run_skill(skill, {}, fake_caller)
    assert recorded == [fake_caller]


def test_step_can_call_caller():
    def llm_step(ctx):
        raw = ctx.caller(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value=raw.choices[0].message.content)

    def fake_caller(**_kw) -> Any:
        return DummyResponse("hello back")

    skill = Skill(name="chat", steps=[Step("llm", llm_step)])
    result = run_skill(skill, {}, fake_caller)
    assert result.value == "hello back"


def test_run_skill_propagates_step_exception():
    def boom(_ctx):
        msg = "step exploded"
        raise RuntimeError(msg)

    skill = Skill(name="err", steps=[Step("boom", boom)])
    with pytest.raises(RuntimeError, match="step exploded"):
        run_skill(skill, {}, _noop)


def test_iter_skill_propagates_and_preserves_prior_yields():
    seen: list[str] = []

    def ok(_ctx):
        return StepResult(value="ok", resolved=False)

    def boom(_ctx):
        msg = "step exploded"
        raise RuntimeError(msg)

    skill = Skill(name="err", steps=[Step("ok", ok), Step("boom", boom)])
    it = iter_skill(skill, {}, _noop)
    name, result = next(it)
    seen.append(name)
    assert result.value == "ok"
    with pytest.raises(RuntimeError, match="step exploded"):
        next(it)
    assert seen == ["ok"]
