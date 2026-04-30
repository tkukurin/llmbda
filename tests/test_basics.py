"""Smoke tests for tk.llmbda core."""

from __future__ import annotations

from typing import Any

import pytest

from tk.llmbda import (
    Skill,
    SkillContext,
    StepResult,
    iter_skill,
    last,
    lm,
    run_skill,
)


def echo_step(ctx: SkillContext) -> StepResult:
    """Step that returns the entry as-is."""
    return StepResult(value=ctx.entry)


def counting_step(ctx: SkillContext) -> StepResult:
    """Step that counts prior steps."""
    return StepResult(value=len(ctx.trace), meta={"seen": list(ctx.trace.keys())})


def _documented_step(_ctx: SkillContext) -> StepResult:
    """Extract a weekday name."""
    return StepResult(value=None)


def _bare_step(_ctx: SkillContext) -> StepResult:
    return StepResult(value="ok")


def test_single_step():
    skill = Skill(name="echo", steps=[Skill("echo", fn=echo_step)])
    trace = run_skill(skill, {"x": 1})
    assert isinstance(trace, dict)
    assert last(trace).value == {"x": 1}
    assert "echo" in trace
    assert trace["echo"].value == {"x": 1}


def test_all_steps_run_in_sequence():
    skill = Skill(
        name="chain",
        steps=[
            Skill("a", fn=counting_step),
            Skill("b", fn=counting_step),
            Skill("c", fn=counting_step),
        ],
    )
    trace = run_skill(skill, {})
    assert last(trace).value == 2
    assert list(trace) == ["a", "b", "c"]


def test_prior_accumulates():
    def deposit(_ctx):
        return StepResult(value="first", meta={"order": 1})

    def check(ctx):
        return StepResult(value=f"saw {ctx.trace['deposit'].value}")

    skill = Skill(
        name="acc",
        steps=[Skill("deposit", fn=deposit), Skill("check", fn=check)],
    )
    trace = run_skill(skill, {})
    assert last(trace).value == "saw first"
    assert trace["deposit"].value == "first"
    assert trace["deposit"].meta == {"order": 1}


@pytest.mark.parametrize(
    "make_skill",
    [
        lambda fn: Skill("dup", steps=[Skill("same", fn=fn), Skill("same", fn=fn)]),
        lambda fn: Skill(
            "s",
            steps=[
                Skill("same", fn=fn, steps=[Skill("child", fn=fn)]),
                Skill("same", fn=fn),
            ],
        ),
        lambda fn: Skill(
            "s",
            steps=[
                Skill(
                    "orch", fn=fn, steps=[Skill("same", fn=fn), Skill("same", fn=fn)]
                ),
            ],
        ),
    ],
    ids=["flat", "outer-trace", "orchestrator-children"],
)
def test_duplicate_names_raise_before_running(make_skill):
    called: list[str] = []

    def _step(_ctx):
        called.append("ran")
        return StepResult(value=None)

    with pytest.raises(ValueError, match="same"):
        list(iter_skill(make_skill(_step), {}))
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
    trace = run_skill(skill, {})
    assert list(trace) == ["same", "orch"]


def test_empty_skill():
    skill = Skill(name="noop")
    trace = run_skill(skill, {"x": 1})
    assert trace == {}
    assert last(trace).value is None


def test_step_meta_preserved():
    def _with_meta(_ctx):
        return StepResult(value=42, meta={"custom": "data", "extra": True})

    skill = Skill(name="meta", steps=[Skill("with_meta", fn=_with_meta)])
    trace = run_skill(skill, {})
    assert last(trace).meta == {"custom": "data", "extra": True}


def test_step_meta_not_mutated_by_runtime():
    emitted = StepResult(value=1, meta={"key": "original"})

    def _emit(_ctx):
        return emitted

    skill = Skill(name="s", steps=[Skill("a", fn=_emit)])
    trace = run_skill(skill, {})
    assert trace["a"] is emitted
    assert emitted.meta == {"key": "original"}


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


@pytest.mark.parametrize(
    ("fn", "kwargs", "expected"),
    [
        (lambda _: StepResult(value=None), {}, ""),
        (_documented_step, {}, "Extract a weekday name."),
        (_documented_step, {"description": "Custom description"}, "Custom description"),
    ],
    ids=["no-docstring", "from-docstring", "explicit-override"],
)
def test_skill_description(fn, kwargs, expected):
    assert Skill("s", fn=fn, **kwargs).description == expected


def test_skill_description_from_lm_wrapped_docstring():
    def fake(**_kw: Any) -> str:
        return ""

    @lm(fake)
    def my_step(_ctx: SkillContext, _call: Any) -> StepResult:
        """LLM step docs."""
        return StepResult(value=None)

    s = Skill("my_step", fn=my_step)
    assert s.description == "LLM step docs."


_SYSTEM_HI = {"role": "system", "content": "be terse"}
_USER_HI = {"role": "user", "content": "hi"}


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("be terse", [[_SYSTEM_HI, _USER_HI]]),
        ("", [[_USER_HI]]),
    ],
    ids=["with-prompt", "no-prompt"],
)
def test_lm_system_prompt_handling(prompt, expected):
    seen: list[list[dict[str, str]]] = []

    def spy(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        seen.append(messages)
        return "ok"

    @lm(spy, system_prompt=prompt)
    def llm_step(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value="done")

    run_skill(Skill(name="s", steps=[Skill("llm", fn=llm_step)]), {})
    assert seen == expected


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
    trace = run_skill(skill, {})
    assert last(trace).value == "hello back"


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
    trace = run_skill(skill, {})
    assert last(trace).value == "spy"
    assert len(captured) == 1
    assert captured[0][0] == {"role": "system", "content": "test prompt"}


def test_lm_rebinds_between_steps():
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

    skill = Skill(name="s", steps=[Skill("a", fn=fn_a), Skill("b", fn=fn_b)])
    run_skill(skill, {})
    assert captured == [
        [{"role": "system", "content": "prompt-a"}, {"role": "user", "content": "a"}],
        [{"role": "system", "content": "prompt-b"}, {"role": "user", "content": "b"}],
    ]


def test_prev_starts_as_empty_step_result():
    def check(ctx: SkillContext) -> StepResult:
        assert ctx.prev.value is None
        return StepResult(value="ok")

    skill = Skill(name="s", steps=[Skill("a", fn=check)])
    run_skill(skill, {})


def test_prev_tracks_previous_step():
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
    def check_get(ctx: SkillContext) -> StepResult:
        assert ctx.trace.get("missing") is None
        return StepResult(value="ok")

    skill = Skill(name="s", steps=[Skill("a", fn=check_get)])
    run_skill(skill, {})


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (lambda _: "hello", "hello"),
        (lambda _: None, None),
    ],
    ids=["string", "none"],
)
def test_raw_return_wrapped(fn, expected):
    skill = Skill(name="s", steps=[Skill("a", fn=fn)])
    trace = run_skill(skill, {})
    assert last(trace).value == expected
    assert isinstance(trace["a"], StepResult)


def test_raw_return_in_chain_updates_prev():
    seen: list[Any] = []

    def second(ctx: SkillContext) -> StepResult:
        seen.append(ctx.prev.value)
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Skill("a", fn=lambda _: 42), Skill("b", fn=second)])
    run_skill(skill, {})
    assert seen == [42]


def test_run_skill_kwargs_entry():
    def read_name(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.entry["name"])

    skill = Skill(name="s", steps=[Skill("a", fn=read_name)])
    assert last(run_skill(skill, name="alice")).value == "alice"


def test_iter_skill_kwargs_entry():
    def read_x(ctx):
        return StepResult(value=ctx.entry["x"])

    skill = Skill(name="s", steps=[Skill("a", fn=read_x)])
    [(_, r)] = list(iter_skill(skill, x=10))
    assert r.value == 10


def test_kwargs_and_positional_entry_raises():
    skill = Skill(name="s", steps=[Skill("a", fn=echo_step)])
    with pytest.raises(TypeError, match="not both"):
        run_skill(skill, {"x": 1}, y=2)


def test_positional_entry_still_works():
    skill = Skill(name="s", steps=[Skill("a", fn=echo_step)])
    assert last(run_skill(skill, {"x": 1})).value == {"x": 1}


@pytest.mark.parametrize(
    ("fn", "expected_name", "expected_value"),
    [
        (_bare_step, "_bare_step", "ok"),
        (lambda _: StepResult(value=1), "<lambda>", 1),
    ],
    ids=["named", "lambda"],
)
def test_bare_callable_wrapping(fn, expected_name, expected_value):
    skill = Skill(name="s", steps=[fn])
    assert isinstance(skill.steps[0], Skill)
    assert skill.steps[0].name == expected_name
    assert last(run_skill(skill, {})).value == expected_value


def test_mixed_skill_and_callable_steps():
    def step_b(_ctx: SkillContext) -> StepResult:
        return StepResult(value="b")

    def step_a(_):
        return StepResult(value="a")

    skill = Skill(name="s", steps=[Skill("a", fn=step_a), step_b])
    trace = run_skill(skill, {})
    assert list(trace) == ["a", "step_b"]
    assert last(trace).value == "b"


def test_bare_callable_gets_description_from_docstring():
    def documented(_ctx: SkillContext) -> StepResult:
        """I have docs."""
        return StepResult(value=None)

    skill = Skill(name="s", steps=[documented])
    assert skill.steps[0].description == "I have docs."


def test_last_of_empty_trace():
    assert last({}).value is None
