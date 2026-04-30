"""Tests for async API: arun_skill, aiter_skill, afst_match, async @lm."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tk.llmbda import (
    Skill,
    SkillContext,
    StepResult,
    afst_match,
    aiter_skill,
    arun_skill,
    last,
    lm,
)


def echo_step(ctx: SkillContext) -> StepResult:
    return StepResult(value=ctx.entry)


def counting_step(ctx: SkillContext) -> StepResult:
    return StepResult(value=len(ctx.trace), meta={"seen": list(ctx.trace.keys())})


class TestArunSkill:
    def test_single_sync_step(self):
        skill = Skill(name="s", steps=[Skill("echo", fn=echo_step)])
        trace = asyncio.run(arun_skill(skill, {"x": 1}))
        assert trace["echo"].value == {"x": 1}

    def test_multi_step_sequence(self):
        skill = Skill(
            name="chain",
            steps=[
                Skill("a", fn=counting_step),
                Skill("b", fn=counting_step),
                Skill("c", fn=counting_step),
            ],
        )
        trace = asyncio.run(arun_skill(skill, {}))
        assert list(trace) == ["a", "b", "c"]
        assert trace["c"].value == 2

    def test_async_step_fn(self):
        async def async_step(ctx: SkillContext) -> StepResult:
            await asyncio.sleep(0)
            return StepResult(value=f"async:{ctx.entry}")

        skill = Skill(name="s", steps=[Skill("a", fn=async_step)])
        trace = asyncio.run(arun_skill(skill, "hello"))
        assert trace["a"].value == "async:hello"

    def test_mixed_sync_and_async_steps(self):
        async def async_upper(ctx: SkillContext) -> StepResult:
            await asyncio.sleep(0)
            return StepResult(value=ctx.entry.upper())

        def sync_len(ctx: SkillContext) -> StepResult:
            return StepResult(value=len(ctx.prev.value))

        skill = Skill(
            name="s",
            steps=[Skill("upper", fn=async_upper), Skill("len", fn=sync_len)],
        )
        trace = asyncio.run(arun_skill(skill, "hello"))
        assert trace["upper"].value == "HELLO"
        assert trace["len"].value == 5

    def test_async_lm_step(self):
        call_log: list[list[dict]] = []

        async def async_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
            call_log.append(messages)
            return "model response"

        @lm(async_model, system_prompt="Be helpful.")
        async def my_step(ctx: SkillContext, call) -> StepResult:
            result = await call(messages=[{"role": "user", "content": ctx.entry}])
            return StepResult(value=result)

        skill = Skill(name="s", steps=[Skill("lm", fn=my_step)])
        trace = asyncio.run(arun_skill(skill, "test"))
        assert trace["lm"].value == "model response"
        assert len(call_log) == 1
        assert call_log[0][0] == {"role": "system", "content": "Be helpful."}
        assert call_log[0][1] == {"role": "user", "content": "test"}

    def test_async_lm_no_system_prompt(self):
        call_log: list[list[dict]] = []

        async def async_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
            call_log.append(messages)
            return "bare"

        @lm(async_model)
        async def step(ctx: SkillContext, call) -> StepResult:
            r = await call(messages=[{"role": "user", "content": ctx.entry}])
            return StepResult(value=r)

        skill = Skill(name="s", steps=[Skill("a", fn=step)])
        trace = asyncio.run(arun_skill(skill, "hi"))
        assert trace["a"].value == "bare"
        assert call_log[0] == [{"role": "user", "content": "hi"}]

    def test_duplicate_names_raise(self):
        skill = Skill(
            name="s",
            steps=[Skill("x", fn=echo_step), Skill("x", fn=echo_step)],
        )
        with pytest.raises(ValueError, match="x"):
            asyncio.run(arun_skill(skill, {}))

    def test_kwargs_entry(self):
        def read_name(ctx: SkillContext) -> StepResult:
            return StepResult(value=ctx.entry["name"])

        skill = Skill(name="s", steps=[Skill("a", fn=read_name)])
        trace = asyncio.run(arun_skill(skill, name="alice"))
        assert last(trace).value == "alice"

    def test_prev_tracks_between_async_steps(self):
        seen_prev: list[Any] = []

        async def record(ctx: SkillContext) -> StepResult:
            seen_prev.append(ctx.prev.value)
            return StepResult(value=len(seen_prev))

        skill = Skill(
            name="s",
            steps=[Skill("a", fn=record), Skill("b", fn=record), Skill("c", fn=record)],
        )
        asyncio.run(arun_skill(skill, {}))
        assert seen_prev == [None, 1, 2]

    def test_async_step_exception_propagates(self):
        async def boom(_ctx: SkillContext) -> StepResult:
            msg = "async boom"
            raise RuntimeError(msg)

        skill = Skill(name="s", steps=[Skill("b", fn=boom)])
        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(arun_skill(skill, {}))

    def test_raw_return_wrapped(self):
        async def raw(_ctx: SkillContext):
            return "just a string"

        skill = Skill(name="s", steps=[Skill("a", fn=raw)])
        trace = asyncio.run(arun_skill(skill, {}))
        assert trace["a"].value == "just a string"
        assert isinstance(trace["a"], StepResult)


class TestAiterSkill:
    def test_yields_each_step(self):
        skill = Skill(
            name="s",
            steps=[Skill("a", fn=counting_step), Skill("b", fn=counting_step)],
        )

        async def collect():
            return [(n, r) async for n, r in aiter_skill(skill, {})]

        pairs = asyncio.run(collect())
        assert [n for n, _ in pairs] == ["a", "b"]
        assert [r.value for _, r in pairs] == [0, 1]

    def test_async_steps_yield(self):
        async def step_a(_ctx: SkillContext) -> StepResult:
            return StepResult(value="A")

        async def step_b(ctx: SkillContext) -> StepResult:
            return StepResult(value=f"B after {ctx.prev.value}")

        skill = Skill(name="s", steps=[Skill("a", fn=step_a), Skill("b", fn=step_b)])

        async def collect():
            return [(n, r) async for n, r in aiter_skill(skill, {})]

        pairs = asyncio.run(collect())
        assert pairs[1][1].value == "B after A"

    def test_break_early(self):
        call_count = [0]

        async def step(_ctx: SkillContext) -> StepResult:
            call_count[0] += 1
            return StepResult(value=call_count[0])

        skill = Skill(
            name="s",
            steps=[Skill("a", fn=step), Skill("b", fn=step), Skill("c", fn=step)],
        )

        async def run():
            results = []
            async for name, _r in aiter_skill(skill, {}):
                results.append(name)
                if name == "a":
                    break
            return results

        assert asyncio.run(run()) == ["a"]
        assert call_count[0] == 1


class TestAfstMatch:
    def test_returns_first_non_none(self):
        def none_step(_ctx: SkillContext) -> StepResult:
            return StepResult(value=None)

        def value_step(_ctx: SkillContext) -> StepResult:
            return StepResult(value="found")

        def never_step(_ctx: SkillContext) -> StepResult:
            msg = "should not run"
            raise AssertionError(msg)

        steps = [
            Skill("a", fn=none_step),
            Skill("b", fn=value_step),
            Skill("c", fn=never_step),
        ]

        async def run():
            ctx = SkillContext(entry="test")
            return await afst_match(ctx, steps)

        result = asyncio.run(run())
        assert result.value == "found"

    def test_returns_empty_when_all_none(self):
        def none_step(_ctx: SkillContext) -> StepResult:
            return StepResult(value=None)

        steps = [Skill("a", fn=none_step), Skill("b", fn=none_step)]

        async def run():
            ctx = SkillContext(entry="x")
            return await afst_match(ctx, steps)

        assert asyncio.run(run()).value is None

    def test_works_with_async_steps(self):
        async def async_none(_ctx: SkillContext) -> StepResult:
            return StepResult(value=None)

        async def async_hit(_ctx: SkillContext) -> StepResult:
            await asyncio.sleep(0)
            return StepResult(value="async hit")

        steps = [Skill("a", fn=async_none), Skill("b", fn=async_hit)]

        async def run():
            ctx = SkillContext(entry="x")
            return await afst_match(ctx, steps)

        assert asyncio.run(run()).value == "async hit"


class TestAsyncLmDecorator:
    def test_introspection_attrs_preserved(self):
        async def fake(*, messages, **_kw):  # noqa: ARG001
            return "x"

        @lm(fake, system_prompt="test prompt")
        async def my_step(_ctx: SkillContext, _call) -> StepResult:
            """My docs."""
            return StepResult(value=None)

        assert my_step.lm_system_prompt == "test prompt"
        assert my_step.lm_model is fake
        assert asyncio.iscoroutinefunction(my_step)

    def test_wrapped_preserves_original(self):
        async def model(*, messages, **_kw):  # noqa: ARG001
            return "x"

        @lm(model)
        async def step(_ctx: SkillContext, _call) -> StepResult:
            """Original fn."""
            return StepResult(value=None)

        assert step.__wrapped__.__doc__ == "Original fn."

    def test_multi_lm_async_steps(self):
        log: list[str] = []

        async def model_a(*, messages, **_kw):  # noqa: ARG001
            log.append("a")
            return "from_a"

        async def model_b(*, messages, **_kw):  # noqa: ARG001
            log.append("b")
            return "from_b"

        @lm(model_a, system_prompt="pa")
        async def step_a(_ctx: SkillContext, call) -> StepResult:
            r = await call(messages=[{"role": "user", "content": "q"}])
            return StepResult(value=r)

        @lm(model_b, system_prompt="pb")
        async def step_b(ctx: SkillContext, call) -> StepResult:
            r = await call(
                messages=[{"role": "user", "content": ctx.prev.value}]
            )
            return StepResult(value=r)

        skill = Skill(name="s", steps=[Skill("a", fn=step_a), Skill("b", fn=step_b)])
        trace = asyncio.run(arun_skill(skill, {}))
        assert trace["a"].value == "from_a"
        assert trace["b"].value == "from_b"
        assert log == ["a", "b"]

    def test_description_from_async_lm_docstring(self):
        async def model(*, messages, **_kw):  # noqa: ARG001
            return ""

        @lm(model)
        async def step(_ctx: SkillContext, _call) -> StepResult:
            """Extract dates from text."""
            return StepResult(value=None)

        s = Skill("step", fn=step)
        assert s.description == "Extract dates from text."
