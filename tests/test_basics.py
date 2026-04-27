"""Smoke tests for tk.llmbda core."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tk.llmbda import (
    ROOT,
    Skill,
    SkillContext,
    SkillResult,
    StepResult,
    aiter_skill,
    iter_skill,
    lm,
)


async def async_echo_step(ctx: SkillContext) -> StepResult:
    """Async step that returns the entry as-is."""
    return StepResult(value=ctx.entry)


def echo_step(ctx: SkillContext) -> StepResult:
    """Step that returns the entry as-is."""
    return StepResult(value=ctx.entry)


def counting_step(ctx: SkillContext) -> StepResult:
    """Step that counts prior steps and falls through."""
    return StepResult(
        value=len(ctx.trace),
        metadata={"seen": list(ctx.trace.keys())},
    )


def test_single_step(run):
    skill = Skill(name="echo", steps=[Skill("echo", fn=echo_step)])
    result = run(skill, {"x": 1})
    assert isinstance(result, SkillResult)
    assert result.value == {"x": 1}
    assert result.skill == "echo"
    assert result.resolved_by == ("echo",)
    assert "echo" in result.trace
    assert result.trace["echo"].value == {"x": 1}


def test_resolved_short_circuits(run):
    def _resolver(_ctx):
        return StepResult(value="stopped", resolved=True)

    def _unreachable(_ctx):
        msg = "should not be called"
        raise AssertionError(msg)

    skill = Skill(
        name="short",
        steps=[Skill("resolver", fn=_resolver), Skill("unreachable", fn=_unreachable)],
    )
    result = run(skill, {})
    assert result.value == "stopped"
    assert result.resolved_by == ("resolver",)
    assert "resolver" in result.trace


def test_implicit_resolved_on_last_step(run):
    skill = Skill(
        name="chain",
        steps=[
            Skill("a", fn=counting_step),
            Skill("b", fn=counting_step),
            Skill("c", fn=counting_step),
        ],
    )
    result = run(skill, {})
    assert result.value == 2  # step c sees a and b
    assert result.resolved_by == ("c",)
    assert list(result.trace) == ["a", "b", "c"]


def test_prior_accumulates(run):
    def deposit(_ctx):
        return StepResult(value="first", metadata={"order": 1})

    def check(ctx):
        prior_val = ctx.trace["deposit"].value
        return StepResult(value=f"saw {prior_val}")

    skill = Skill(
        name="acc",
        steps=[Skill("deposit", fn=deposit), Skill("check", fn=check)],
    )
    result = run(skill, {})
    assert result.value == "saw first"
    assert result.trace["deposit"].value == "first"
    assert result.trace["deposit"].metadata == {"order": 1}


def test_duplicate_step_names_raise_before_running(iter_collect):
    called: list[str] = []

    def _step(_ctx):
        called.append("ran")
        return StepResult(value=None)

    skill = Skill(
        name="dup",
        steps=[Skill("same", fn=_step), Skill("same", fn=_step)],
    )
    with pytest.raises(ValueError, match="same"):
        iter_collect(skill, {})
    assert called == []


def test_duplicate_outer_trace_names_raise_before_running(iter_collect):
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
        iter_collect(skill, {})
    assert called == []


def test_duplicate_orchestrator_child_names_raise_before_running(iter_collect):
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
        iter_collect(skill, {})
    assert called == []


def test_same_name_allowed_across_outer_and_orchestrator_child_scopes(run):
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
    result = run(skill, {})
    assert result.resolved_by == ("orch",)
    assert list(result.trace) == ["same", "orch"]


def test_empty_skill(run):
    skill = Skill(name="noop")
    result = run(skill, {"x": 1})
    assert result.value is None
    assert result.skill == "noop"
    assert result.resolved_by == ()
    assert result.trace == {}


def test_step_metadata_preserved_unchanged(run):
    def _with_meta(_ctx):
        return StepResult(value=42, metadata={"custom": "data", "extra": True})

    skill = Skill(name="meta", steps=[Skill("with_meta", fn=_with_meta)])
    result = run(skill, {})
    assert result.metadata == {"custom": "data", "extra": True}
    assert result.skill == "meta"
    assert result.resolved_by == ("with_meta",)


def test_step_metadata_not_mutated_by_runtime(run):
    """Runtime attrs live on SkillResult; step's metadata dict is untouched."""
    emitted = StepResult(value=1, metadata={"skill": "hijacked"})

    def _clashing(_ctx):
        return emitted

    skill = Skill(name="real", steps=[Skill("clash", fn=_clashing)])
    result = run(skill, {})
    assert result.skill == "real"
    assert result.resolved_by == ("clash",)
    assert emitted.metadata == {"skill": "hijacked"}  # unchanged
    assert result.metadata == {"skill": "hijacked"}


def test_iter_yields_each_step_in_order(iter_collect):
    skill = Skill(
        name="chain",
        steps=[
            Skill("a", fn=counting_step),
            Skill("b", fn=counting_step),
            Skill("c", fn=counting_step),
        ],
    )
    yielded = iter_collect(skill, {})
    assert [name for name, _ in yielded] == ["a", "b", "c"]
    assert [r.value for _, r in yielded] == [0, 1, 2]


def test_iter_stops_after_resolved(iter_collect):
    def _stop(_ctx):
        return StepResult(value="done", resolved=True)

    def _never(_ctx):
        msg = "should not run"
        raise AssertionError(msg)

    skill = Skill(
        name="s",
        steps=[Skill("stop", fn=_stop), Skill("never", fn=_never)],
    )
    yielded = iter_collect(skill, {})
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


def test_iter_collects_into_dict(iter_collect):
    skill = Skill(
        name="chain",
        steps=[Skill("a", fn=counting_step), Skill("b", fn=counting_step)],
    )
    trace = dict(iter_collect(skill, {}))
    assert set(trace) == {"a", "b"}
    assert trace["b"].value == 1


def test_iter_empty_skill_yields_nothing(iter_collect):
    assert iter_collect(Skill(name="noop"), {}) == []


def test_run_skill_propagates_step_exception(run):
    def boom(_ctx):
        msg = "step exploded"
        raise RuntimeError(msg)

    skill = Skill(name="err", steps=[Skill("boom", fn=boom)])
    with pytest.raises(RuntimeError, match="step exploded"):
        run(skill, {})


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


def test_lm_prepends_system_prompt(run):
    seen: list[list[dict[str, str]]] = []

    def spy(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        seen.append(messages)
        return "ok"

    @lm(spy, system_prompt="be terse")
    def llm_step(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Skill("llm", fn=llm_step)])
    run(skill, {})
    assert seen == [
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    ]


def test_lm_no_prompt_passes_through(run):
    seen: list[list[dict[str, str]]] = []

    def spy(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        seen.append(messages)
        return "ok"

    @lm(spy)
    def llm_step(_ctx: SkillContext, call: Any) -> StepResult:
        call(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value="done")

    skill = Skill(name="s", steps=[Skill("llm", fn=llm_step)])
    run(skill, {})
    assert seen == [[{"role": "user", "content": "hi"}]]


def test_lm_per_step_model(run):
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
    run(skill, {})
    assert captured == ["a", "b"]


def test_lm_step_receives_bound_caller(run):
    def fake(**_kw: Any) -> str:
        return "hello back"

    @lm(fake)
    def llm_step(_ctx: SkillContext, call: Any) -> StepResult:
        raw = call(messages=[{"role": "user", "content": "hi"}])
        return StepResult(value=raw)

    skill = Skill(name="chat", steps=[Skill("llm", fn=llm_step)])
    result = run(skill, {})
    assert result.value == "hello back"


def test_lm_introspection_attrs():
    def fake(**_kw: Any) -> str:
        return ""

    @lm(fake, system_prompt="be terse")
    def llm_step(_ctx: SkillContext, _call: Any) -> StepResult:
        return StepResult(value=None)

    assert llm_step.lm_system_prompt == "be terse"  # type: ignore[attr-defined]
    assert llm_step.lm_model is fake  # type: ignore[attr-defined]


def test_lm_rewrap_for_testing(run):
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
    result = run(skill, {})
    assert result.value == "spy"
    assert len(captured) == 1
    assert captured[0][0] == {"role": "system", "content": "test prompt"}


def test_lm_rebinds_between_steps(run):
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
    run(skill, {})
    assert captured == [
        [{"role": "system", "content": "prompt-a"}, {"role": "user", "content": "a"}],
        [{"role": "system", "content": "prompt-b"}, {"role": "user", "content": "b"}],
    ]


def test_resolved_defaults_to_false():
    r = StepResult(value=42)
    assert r.resolved is False


def test_prev_is_root_initially(run):
    def check_root(ctx: SkillContext) -> StepResult:
        assert ctx.prev is ROOT
        return StepResult(value="ok")

    skill = Skill(name="s", steps=[Skill("a", fn=check_root)])
    run(skill, {})


def test_prev_tracks_previous_step(run):
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
    run(skill, {})
    assert seen_prev == [None, 1, 2]


def test_prior_keyerror_includes_available_steps(run):
    def bad_step(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["nonexistent"])

    skill = Skill(
        name="s",
        steps=[Skill("a", fn=lambda _: StepResult(value=1)), Skill("b", fn=bad_step)],
    )
    with pytest.raises(KeyError, match=r"nonexistent.*available.*\ba\b"):
        run(skill, {})


def test_prior_get_returns_none_for_missing(run):
    """dict.get bypasses __missing__ and returns the default."""
    def check_get(ctx: SkillContext) -> StepResult:
        assert ctx.trace.get("missing") is None
        return StepResult(value="ok")

    skill = Skill(name="s", steps=[Skill("a", fn=check_get)])
    run(skill, {})


def test_async_step_fn(run):
    """Native async def step works through both sync and async paths."""
    skill = Skill(name="s", steps=[Skill("echo", fn=async_echo_step)])
    result = run(skill, {"x": 42})
    assert result.value == {"x": 42}
    assert result.resolved_by == ("echo",)


def test_async_step_with_prev(run):
    """Async step can read ctx.prev set by a prior sync step."""
    def sync_first(_ctx: SkillContext) -> StepResult:
        return StepResult(value=10)

    async def async_second(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.prev.value + 1)

    skill = Skill(
        name="s",
        steps=[Skill("a", fn=sync_first), Skill("b", fn=async_second)],
    )
    result = run(skill, {})
    assert result.value == 11


def test_streaming_step_yields_chunks_then_result():
    """Async generator step yields str chunks followed by a StepResult."""
    async def streaming(ctx: SkillContext) -> StepResult:
        for word in ctx.entry["text"].split():
            yield word
        yield StepResult(value=ctx.entry["text"].upper())

    skill = Skill(name="s", steps=[Skill("stream", fn=streaming)])
    collected = asyncio.run(_acollect(skill, {"text": "hello world"}))
    assert collected == [
        ("stream", "hello"),
        ("stream", "world"),
        ("stream", StepResult(value="HELLO WORLD")),
    ]


def test_streaming_step_result_in_trace(run):
    """The final StepResult from a streaming step lands in the trace."""
    async def streaming(_ctx: SkillContext) -> StepResult:
        yield "chunk"
        yield StepResult(value="final")

    def after(ctx: SkillContext) -> StepResult:
        return StepResult(value=f"after:{ctx.trace['stream'].value}")

    skill = Skill(name="s", steps=[Skill("stream", fn=streaming), Skill("after", fn=after)])
    result = run(skill, {})
    assert result.value == "after:final"
    assert result.trace["stream"].value == "final"


def test_streaming_step_prev_tracks(run):
    """ctx.prev is set to the StepResult from a streaming step."""
    async def streaming(_ctx: SkillContext) -> StepResult:
        yield "a"
        yield StepResult(value=42)

    def check_prev(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.prev.value + 1)

    skill = Skill(name="s", steps=[Skill("stream", fn=streaming), Skill("check", fn=check_prev)])
    result = run(skill, {})
    assert result.value == 43


def test_streaming_step_missing_step_result_raises():
    """Async generator that never yields a StepResult raises TypeError."""
    async def bad(_ctx: SkillContext) -> StepResult:
        yield "chunk"

    skill = Skill(name="s", steps=[Skill("bad", fn=bad)])
    with pytest.raises(TypeError, match="streaming step must yield a final StepResult"):
        asyncio.run(_arun(skill, {}))


def test_streaming_step_resolved_short_circuits(run):
    """A streaming step with resolved=True stops remaining steps."""
    async def streaming(_ctx: SkillContext) -> StepResult:
        yield "working..."
        yield StepResult(value="done", resolved=True)

    def unreachable(_ctx: SkillContext) -> StepResult:
        raise AssertionError("should not run")

    skill = Skill(name="s", steps=[Skill("stream", fn=streaming), Skill("never", fn=unreachable)])
    result = run(skill, {})
    assert result.value == "done"
    assert result.resolved_by == ("stream",)


def test_parallel_steps_run_concurrently(run):
    """Parallel composite runs children independently and merges trace."""
    async def slow_a(_ctx: SkillContext) -> StepResult:
        return StepResult(value="a")

    async def slow_b(_ctx: SkillContext) -> StepResult:
        return StepResult(value="b")

    skill = Skill(
        name="s",
        steps=[
            Skill("par", parallel=True, steps=[
                Skill("a", fn=slow_a),
                Skill("b", fn=slow_b),
            ]),
        ],
    )
    result = run(skill, {})
    assert result.trace["a"].value == "a"
    assert result.trace["b"].value == "b"


def test_parallel_steps_get_independent_contexts(run):
    """Parallel children don't see each other's trace."""
    def first(_ctx: SkillContext) -> StepResult:
        return StepResult(value=1)

    def second(ctx: SkillContext) -> StepResult:
        assert ctx.trace.get("first") is None
        return StepResult(value=2)

    skill = Skill(
        name="s",
        steps=[
            Skill("par", parallel=True, steps=[
                Skill("first", fn=first),
                Skill("second", fn=second),
            ]),
        ],
    )
    result = run(skill, {})
    assert result.trace["first"].value == 1
    assert result.trace["second"].value == 2


def test_parallel_steps_prev_set_to_last_child(run):
    """After a parallel block, ctx.prev is the last child's result."""
    skill = Skill(
        name="s",
        steps=[
            Skill("par", parallel=True, steps=[
                Skill("a", fn=lambda _: StepResult(value="a")),
                Skill("b", fn=lambda _: StepResult(value="b")),
            ]),
            Skill("after", fn=lambda ctx: StepResult(value=f"prev:{ctx.prev.value}")),
        ],
    )
    result = run(skill, {})
    assert result.value == "prev:b"


def test_parallel_after_sequential_shares_trace(run):
    """A parallel block after a sequential step shares the prior trace."""
    def setup(_ctx: SkillContext) -> StepResult:
        return StepResult(value="setup")

    def par_child(ctx: SkillContext) -> StepResult:
        prior = ctx.trace.get("setup")
        return StepResult(value=f"saw:{prior.value if prior else None}")

    skill = Skill(
        name="s",
        steps=[
            Skill("setup", fn=setup),
            Skill("par", parallel=True, steps=[
                Skill("child", fn=par_child),
            ]),
        ],
    )
    result = run(skill, {})
    assert result.trace["child"].value == "saw:setup"


def test_parallel_mixed_sync_async(run):
    """Parallel block handles a mix of sync and async step fns."""
    def sync_step(_ctx: SkillContext) -> StepResult:
        return StepResult(value="sync")

    async def async_step(_ctx: SkillContext) -> StepResult:
        return StepResult(value="async")

    skill = Skill(
        name="s",
        steps=[
            Skill("par", parallel=True, steps=[
                Skill("s", fn=sync_step),
                Skill("a", fn=async_step),
            ]),
        ],
    )
    result = run(skill, {})
    assert result.trace["s"].value == "sync"
    assert result.trace["a"].value == "async"


async def _acollect(skill, entry):
    return [item async for item in aiter_skill(skill, entry)]


async def _arun(skill, entry):
    from tk.llmbda import arun_skill
    return await arun_skill(skill, entry)