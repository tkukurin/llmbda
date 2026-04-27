"""Smoke tests for tk.llmbda core."""

from __future__ import annotations

from typing import Any

import pytest

from tk.llmbda import (
    ROOT,
    Skill,
    SkillContext,
    SkillResult,
    StepResult,
    iter_skill,
    lm,
    run_skill,
)


def echo_step(ctx: SkillContext) -> StepResult:
    """Step that returns the entry as-is."""
    return StepResult(value=ctx.entry)


def counting_step(ctx: SkillContext) -> StepResult:
    """Step that counts prior steps and falls through."""
    return StepResult(
        value=len(ctx.trace),
        metadata={"seen": list(ctx.trace.keys())},
    )


def test_single_step():
    skill = Skill(name="echo", steps=[Skill("echo", fn=echo_step)])
    result = run_skill(skill, {"x": 1})
    assert isinstance(result, SkillResult)
    assert result.value == {"x": 1}
    assert result.skill == "echo"
    assert result.resolved_by == ("echo",)
    assert "echo" in result.trace
    assert result.trace["echo"].value == {"x": 1}


def test_resolved_short_circuits():
    def _resolver(_ctx):
        return StepResult(value="stopped", resolved=True)

    def _unreachable(_ctx):
        msg = "should not be called"
        raise AssertionError(msg)

    skill = Skill(
        name="short",
        steps=[Skill("resolver", fn=_resolver), Skill("unreachable", fn=_unreachable)],
    )
    result = run_skill(skill, {})
    assert result.value == "stopped"
    assert result.resolved_by == ("resolver",)
    assert "resolver" in result.trace


def test_implicit_resolved_on_last_step():
    skill = Skill(
        name="chain",
        steps=[
            Skill("a", fn=counting_step),
            Skill("b", fn=counting_step),
            Skill("c", fn=counting_step),
        ],
    )
    result = run_skill(skill, {})
    assert result.value == 2  # step c sees a and b
    assert result.resolved_by == ("c",)
    assert list(result.trace) == ["a", "b", "c"]


def test_prior_accumulates():
    def deposit(_ctx):
        return StepResult(value="first", metadata={"order": 1})

    def check(ctx):
        prior_val = ctx.trace["deposit"].value
        return StepResult(value=f"saw {prior_val}")

    skill = Skill(
        name="acc",
        steps=[Skill("deposit", fn=deposit), Skill("check", fn=check)],
    )
    result = run_skill(skill, {})
    assert result.value == "saw first"
    assert result.trace["deposit"].value == "first"
    assert result.trace["deposit"].metadata == {"order": 1}


def test_duplicate_step_names_raise_before_running():
    called: list[str] = []

    def _step(_ctx):
        called.append("ran")
        return StepResult(value=None)

    skill = Skill(
        name="dup",
        steps=[Skill("same", fn=_step), Skill("same", fn=_step)],
    )
    with pytest.raises(ValueError, match="same"):
        list(iter_skill(skill, {}))
    assert called == []


def test_duplicate_outer_trace_names_raise_before_running():
    called: list[str] = []

    def _step(_ctx):
        called.append("ran")
        return StepResult(value=None)

    skill = Skill(
        name="s",
        steps=[
            Skill("same", fn=_step, steps=[Skill("child", fn=_step)]),
            Skill("same", fn=_step),
        ],
    )
    with pytest.raises(ValueError, match="same"):
        list(iter_skill(skill, {}))
    assert called == []


def test_duplicate_orchestrator_child_names_raise_before_running():
    called: list[str] = []

    def _step(_ctx):
        called.append("ran")
        return StepResult(value=None)

    skill = Skill(
        name="s",
        steps=[
            Skill(
                "orch",
                fn=_step,
                steps=[Skill("same", fn=_step), Skill("same", fn=_step)],
            ),
        ],
    )
    with pytest.raises(ValueError, match="same"):
        list(iter_skill(skill, {}))
    assert called == []


def test_same_name_allowed_across_outer_and_orchestrator_child_scopes():
    def _step(_ctx):
        return StepResult(value="ok")

    def _orch_step(_ctx, _steps):
        return StepResult(value="ok")

    skill = Skill(
        name="s",
        steps=[
            Skill("same", fn=_step),
            Skill("orch", fn=_orch_step, steps=[Skill("same", fn=_step)]),
        ],
    )
    result = run_skill(skill, {})
    assert result.resolved_by == ("orch",)
    assert list(result.trace) == ["same", "orch"]


def test_empty_skill():
    skill = Skill(name="noop")
    result = run_skill(skill, {"x": 1})
    assert result.value is None
    assert result.skill == "noop"
    assert result.resolved_by == ()
    assert result.trace == {}


def test_step_metadata_preserved_unchanged():
    def _with_meta(_ctx):
        return StepResult(value=42, metadata={"custom": "data", "extra": True})

    skill = Skill(name="meta", steps=[Skill("with_meta", fn=_with_meta)])
    result = run_skill(skill, {})
    assert result.metadata == {"custom": "data", "extra": True}
    assert result.skill == "meta"
    assert result.resolved_by == ("with_meta",)


def test_step_metadata_not_mutated_by_runtime():
    """Runtime attrs live on SkillResult; step's metadata dict is untouched."""
    emitted = StepResult(value=1, metadata={"skill": "hijacked"})

    def _clashing(_ctx):
        return emitted

    skill = Skill(name="real", steps=[Skill("clash", fn=_clashing)])
    result = run_skill(skill, {})
    assert result.skill == "real"
    assert result.resolved_by == ("clash",)
    assert emitted.metadata == {"skill": "hijacked"}  # unchanged
    assert result.metadata == {"skill": "hijacked"}


def test_iter_yields_each_step_in_order():
    skill = Skill(
        name="chain",
        steps=[
            Skill("a", fn=counting_step),
            Skill("b", fn=counting_step),
            Skill("c", fn=counting_step),
        ],
    )
    yielded = list(iter_skill(skill, {}))
    assert [name for name, _ in yielded] == ["a", "b", "c"]
    assert [r.value for _, r in yielded] == [0, 1, 2]


def test_iter_stops_after_resolved():
    def _stop(_ctx):
        return StepResult(value="done", resolved=True)

    def _never(_ctx):
        msg = "should not run"
        raise AssertionError(msg)

    skill = Skill(
        name="s",
        steps=[Skill("stop", fn=_stop), Skill("never", fn=_never)],
    )
    yielded = list(iter_skill(skill, {}))
    assert [name for name, _ in yielded] == ["stop"]


def test_iter_stop_decision_survives_result_mutation():
    called: list[str] = []

    def _stop(_ctx):
        return StepResult(value="done", resolved=True)

    def _never(_ctx):
        called.append("never")
        return StepResult(value="unreachable")

    skill = Skill(
        name="s",
        steps=[Skill("stop", fn=_stop), Skill("never", fn=_never)],
    )
    it = iter_skill(skill, {})
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
            Skill("a", fn=counting_step),
            Skill("b", fn=counting_step),
            Skill("c", fn=counting_step),
        ],
    )
    for name, _ in iter_skill(skill, {}):
        seen.append(name)
        if name == "a":
            break
    assert seen == ["a"]


def test_iter_collects_into_dict():
    skill = Skill(
        name="chain",
        steps=[Skill("a", fn=counting_step), Skill("b", fn=counting_step)],
    )
    trace = dict(iter_skill(skill, {}))
    assert set(trace) == {"a", "b"}
    assert trace["b"].value == 1


def test_iter_empty_skill_yields_nothing():
    assert list(iter_skill(Skill(name="noop"), {})) == []


def test_run_skill_propagates_step_exception():
    def boom(_ctx):
        msg = "step exploded"
        raise RuntimeError(msg)

    skill = Skill(name="err", steps=[Skill("boom", fn=boom)])
    with pytest.raises(RuntimeError, match="step exploded"):
        run_skill(skill, {})


def test_iter_skill_propagates_and_preserves_prior_yields():
    seen: list[str] = []

    def ok(_ctx):
        return StepResult(value="ok")

    def boom(_ctx):
        msg = "step exploded"
        raise RuntimeError(msg)

    skill = Skill(name="err", steps=[Skill("ok", fn=ok), Skill("boom", fn=boom)])
    it = iter_skill(skill, {})
    name, result = next(it)
    seen.append(name)
    assert result.value == "ok"
    with pytest.raises(RuntimeError, match="step exploded"):
        next(it)
    assert seen == ["ok"]


def test_skill_description_from_docstring():
    def my_step(_ctx: SkillContext) -> StepResult:
        """Extract a weekday name."""
        return StepResult(value=None)

    s = Skill("my_step", fn=my_step)
    assert s.description == "Extract a weekday name."


def test_skill_description_explicit_override():
    def my_step(_ctx: SkillContext) -> StepResult:
        """This docstring is ignored."""
        return StepResult(value=None)

    s = Skill("my_step", fn=my_step, description="Custom description")
    assert s.description == "Custom description"


def test_skill_description_no_docstring():
    s = Skill("anon", fn=lambda _ctx: StepResult(value=None))
    assert s.description == ""


def test_skill_description_from_lm_wrapped_docstring():
    def fake(**_kw: Any) -> str:
        return ""

    @lm(fake)
    def my_step(_ctx: SkillContext, _call: Any) -> StepResult:
        """LLM step docs."""
        return StepResult(value=None)

    s = Skill("my_step", fn=my_step)
    assert s.description == "LLM step docs."


def test_lm_prepends_system_prompt():
    seen: list[list[dict[str, str]]] = []

    def spy(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        seen.append(messages)
        return "ok"

    @lm(spy, system_prompt="be terse")
    def llm_step(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Skill("llm", fn=llm_step)])
    run_skill(skill, {})
    assert seen == [
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    ]


def test_lm_no_prompt_passes_through():
    seen: list[list[dict[str, str]]] = []

    def spy(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        seen.append(messages)
        return "ok"

    @lm(spy)
    def llm_step(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Skill("llm", fn=llm_step)])
    run_skill(skill, {})
    assert seen == [[{"role": "user", "content": "hi"}]]


def test_lm_per_step_model():
    captured: list[str] = []

    def model_a(**_kw: Any) -> str:
        captured.append("a")
        return "a"

    def model_b(**_kw: Any) -> str:
        captured.append("b")
        return "b"

    @lm(model_a, system_prompt="prompt-a")
    def step_a(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "a"}])
        return StepResult(value=None)

    @lm(model_b, system_prompt="prompt-b")
    def step_b(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "b"}])
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Skill("a", fn=step_a), Skill("b", fn=step_b)])
    run_skill(skill, {})
    assert captured == ["a", "b"]


def test_lm_step_receives_bound_caller():
    def fake(**_kw: Any) -> str:
        return "hello back"

    @lm(fake)
    def llm_step(_ctx: SkillContext, call: Any) -> StepResult:
        raw = call(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value=raw)

    skill = Skill(name="chat", steps=[Skill("llm", fn=llm_step)])
    result = run_skill(skill, {})
    assert result.value == "hello back"


def test_lm_introspection_attrs():
    def fake(**_kw: Any) -> str:
        return ""

    @lm(fake, system_prompt="be terse")
    def llm_step(_ctx: SkillContext, _call: Any) -> StepResult:
        return StepResult(value=None)

    assert llm_step.lm_system_prompt == "be terse"  # type: ignore[attr-defined]
    assert llm_step.lm_model is fake  # type: ignore[attr-defined]


def test_lm_rewrap_for_testing():
    def original_model(**_kw: Any) -> str:
        return "original"

    @lm(original_model)
    def llm_step(_ctx: SkillContext, call: Any) -> StepResult:
        """My LLM step."""
        return StepResult(value=call(messages=[{"role": "user", "content": "x"}]))

    captured: list[list[dict[str, str]]] = []

    def spy_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        captured.append(messages)
        return "spy"

    rewrapped = lm(spy_model, system_prompt="test prompt")(llm_step.__wrapped__)  # type: ignore[attr-defined]
    skill = Skill(name="s", steps=[Skill("llm", fn=rewrapped)])
    result = run_skill(skill, {})
    assert result.value == "spy"
    assert len(captured) == 1
    assert captured[0][0] == {"role": "system", "content": "test prompt"}


def test_lm_rebinds_between_steps():
    """Each step sees a caller bound to its own model and prompt."""
    captured: list[list[dict[str, str]]] = []

    def capturing(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        captured.append(messages)
        return "ok"

    @lm(capturing, system_prompt="prompt-a")
    def fn_a(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "a"}])
        return StepResult(value=None)

    @lm(capturing, system_prompt="prompt-b")
    def fn_b(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "b"}])
        return StepResult(value="done")

    skill = Skill(
        name="s",
        steps=[Skill("a", fn=fn_a), Skill("b", fn=fn_b)],
    )
    run_skill(skill, {})
    assert captured == [
        [{"role": "system", "content": "prompt-a"}, {"role": "user", "content": "a"}],
        [{"role": "system", "content": "prompt-b"}, {"role": "user", "content": "b"}],
    ]


def test_resolved_defaults_to_false():
    r = StepResult(value=42)
    assert r.resolved is False


def test_prev_is_root_initially():
    def check_root(ctx: SkillContext) -> StepResult:
        assert ctx.prev is ROOT
        return StepResult(value="ok")

    skill = Skill(name="s", steps=[Skill("a", fn=check_root)])
    run_skill(skill, {})


def test_prev_tracks_previous_step():
    """ctx.prev holds the most recently executed step's result."""
    seen_prev: list[Any] = []

    def record_prev(ctx: SkillContext) -> StepResult:
        seen_prev.append(ctx.prev.value)
        return StepResult(value=len(seen_prev))

    skill = Skill(
        name="s",
        steps=[
            Skill("a", fn=record_prev),
            Skill("b", fn=record_prev),
            Skill("c", fn=record_prev),
        ],
    )
    run_skill(skill, {})
    assert seen_prev == [None, 1, 2]


def test_prior_keyerror_includes_available_steps():
    def bad_step(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["nonexistent"])

    skill = Skill(
        name="s",
        steps=[Skill("a", fn=lambda _: StepResult(value=1)), Skill("b", fn=bad_step)],
    )
    with pytest.raises(KeyError, match=r"nonexistent.*available.*\ba\b"):
        run_skill(skill, {})


def test_prior_get_returns_none_for_missing():
    """dict.get bypasses __missing__ and returns the default."""

    def check_get(ctx: SkillContext) -> StepResult:
        assert ctx.trace.get("missing") is None
        return StepResult(value="ok")

    skill = Skill(name="s", steps=[Skill("a", fn=check_get)])
    run_skill(skill, {})


def test_raw_return_wrapped_as_step_result():
    skill = Skill(name="s", steps=[Skill("a", fn=lambda _: "hello")])
    result = run_skill(skill, {})
    assert result.value == "hello"
    assert result.resolved_by == ("a",)


def test_raw_return_none_wrapped():
    skill = Skill(name="s", steps=[Skill("a", fn=lambda _: None)])
    result = run_skill(skill, {})
    assert result.value is None
    assert "a" in result.trace
    assert isinstance(result.trace["a"], StepResult)


def test_raw_return_in_chain_updates_prev():
    seen: list[Any] = []

    def second(ctx: SkillContext) -> StepResult:
        seen.append(ctx.prev.value)
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Skill("a", fn=lambda _: 42), Skill("b", fn=second)])
    run_skill(skill, {})
    assert seen == [42]


def test_explicit_step_result_still_works():
    """Ensure explicit StepResult isn't double-wrapped."""
    skill = Skill(name="s", steps=[Skill("a", fn=lambda _: StepResult(value="x", resolved=True))])
    result = run_skill(skill, {})
    assert result.value == "x"
    assert result.resolved_by == ("a",)


def test_run_skill_kwargs_entry():
    def read_name(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.entry["name"])

    skill = Skill(name="s", steps=[Skill("a", fn=read_name)])
    assert run_skill(skill, name="alice").value == "alice"


def test_iter_skill_kwargs_entry():
    skill = Skill(name="s", steps=[Skill("a", fn=lambda ctx: StepResult(value=ctx.entry["x"]))])
    [(_, r)] = list(iter_skill(skill, x=10))
    assert r.value == 10


def test_kwargs_and_positional_entry_raises():
    skill = Skill(name="s", steps=[Skill("a", fn=echo_step)])
    with pytest.raises(TypeError, match="not both"):
        run_skill(skill, {"x": 1}, y=2)


def test_positional_entry_still_works():
    skill = Skill(name="s", steps=[Skill("a", fn=echo_step)])
    assert run_skill(skill, {"x": 1}).value == {"x": 1}
