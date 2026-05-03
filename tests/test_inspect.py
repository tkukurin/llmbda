"""Tests for tk.llmbda.inspect adapters."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target, match
from inspect_ai.solver import TaskState

from tk.llmbda import Skill, SkillContext, StepResult
from tk.llmbda.inspect import DEFAULT_TRACE_KEY, skill_solver, step_check, step_scorer


def _make_state(input_text: str = "hello", completion: str = "") -> TaskState:
    return TaskState(
        model="test/model",
        sample_id=1,
        epoch=0,
        input=input_text,
        messages=[],
        output=ModelOutput.from_content(model="test/model", content=completion),
    )


def _identifiers_step(ctx: SkillContext) -> StepResult:
    return StepResult(value=["ID-1", "ID-2"], meta={"raw": ctx.entry})


def _draft_step(ctx: SkillContext) -> StepResult:
    ids = ctx.trace["identifiers"].value
    return StepResult(value=f"Draft mentioning {ids[0]}: answer is C")


def _skill() -> Skill:
    return Skill(
        name="triage",
        steps=[
            Skill(name="identifiers", fn=_identifiers_step),
            Skill(name="draft", fn=_draft_step),
        ],
    )


def test_skill_solver_sets_completion_and_trace():
    solver = skill_solver(_skill())
    state = _make_state(input_text="ticket body")
    out = asyncio.run(solver(state, None))

    assert "answer is C" in out.output.completion
    trace = out.metadata[DEFAULT_TRACE_KEY]
    assert set(trace) == {"identifiers", "draft"}
    assert trace["identifiers"].value == ["ID-1", "ID-2"]
    assert trace["identifiers"].meta == {"raw": "ticket body"}


def test_skill_solver_custom_entry_extractor():
    def _read_ticket(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.entry["body"])

    skill = Skill(name="s", steps=[Skill(name="read", fn=_read_ticket)])
    solver = skill_solver(skill, entry=lambda s: s.metadata["ticket"])

    state = _make_state()
    state.metadata["ticket"] = {"body": "urgent"}
    out = asyncio.run(solver(state, None))

    assert out.output.completion == "urgent"


def test_step_scorer_reads_named_step():
    solver = skill_solver(_skill())
    state = asyncio.run(solver(_make_state(), None))

    scorer = step_scorer("draft", match(location="any"))
    score = asyncio.run(scorer(state, Target("C")))

    assert score.value == "C"


def test_step_scorer_missing_step_raises():
    solver = skill_solver(_skill())
    state = asyncio.run(solver(_make_state(), None))

    scorer = step_scorer("nonexistent", match())
    with pytest.raises(KeyError) as exc:
        asyncio.run(scorer(state, Target("C")))

    msg = str(exc.value)
    assert "nonexistent" in msg
    assert "identifiers" in msg
    assert "draft" in msg


def test_step_scorer_missing_trace_metadata_raises():
    state = _make_state(completion="whatever")
    scorer = step_scorer("draft", match())
    with pytest.raises(KeyError) as exc:
        asyncio.run(scorer(state, Target("C")))
    assert "(none)" in str(exc.value)


def test_step_scorer_custom_projector():
    def _dict_step(_ctx: SkillContext) -> StepResult:
        return StepResult(value={"label": "C", "confidence": 0.9})

    skill = Skill(name="s", steps=[Skill(name="classify", fn=_dict_step)])
    solver = skill_solver(skill)
    state = asyncio.run(solver(_make_state(), None))

    scorer = step_scorer("classify", match(location="any"), project=json.dumps)
    score = asyncio.run(scorer(state, Target("C")))

    assert score.value == "C"


def test_step_scorer_does_not_mutate_caller_state():
    solver = skill_solver(_skill())
    state = asyncio.run(solver(_make_state(), None))
    original_completion = state.output.completion

    scorer = step_scorer("identifiers", match(location="any"), project=json.dumps)
    asyncio.run(scorer(state, Target("ID-1")))

    assert state.output.completion == original_completion


def test_step_check_bool_predicate():
    def _valid_step(_ctx: SkillContext) -> StepResult:
        return StepResult(value="ok", meta={"valid": True})

    skill = Skill(name="s", steps=[Skill(name="check", fn=_valid_step)])
    solver = skill_solver(skill)
    state = asyncio.run(solver(_make_state(), None))

    checker = step_check("check", lambda r: r.meta["valid"])
    score = asyncio.run(checker(state, Target("ignored")))
    assert score.value == 1.0


def test_step_check_false_predicate():
    def _invalid_step(_ctx: SkillContext) -> StepResult:
        return StepResult(value="bad", meta={"valid": False, "errors": ["oops"]})

    skill = Skill(name="s", steps=[Skill(name="v", fn=_invalid_step)])
    solver = skill_solver(skill)
    state = asyncio.run(solver(_make_state(), None))

    checker = step_check("v", lambda r: r.meta["valid"])
    score = asyncio.run(checker(state, Target("ignored")))
    assert score.value == 0.0


def test_step_check_float_predicate():
    def _partial_step(_ctx: SkillContext) -> StepResult:
        return StepResult(value="x", meta={"confidence": 0.75})

    skill = Skill(name="s", steps=[Skill(name="p", fn=_partial_step)])
    solver = skill_solver(skill)
    state = asyncio.run(solver(_make_state(), None))

    checker = step_check("p", lambda r: r.meta["confidence"])
    score = asyncio.run(checker(state, Target("ignored")))
    assert score.value == 0.75


def test_step_check_score_passthrough():
    from inspect_ai.scorer import Score  # noqa: PLC0415

    def _step(_ctx: SkillContext) -> StepResult:
        return StepResult(value="v", meta={"valid": True})

    skill = Skill(name="s", steps=[Skill(name="s", fn=_step)])
    solver = skill_solver(skill)
    state = asyncio.run(solver(_make_state(), None))

    checker = step_check(
        "s", lambda _r: Score(value=1.0, answer="yes", explanation="all good")
    )
    score = asyncio.run(checker(state, Target("ignored")))
    assert score.value == 1.0
    assert score.answer == "yes"


def test_step_check_missing_step_raises():
    solver = skill_solver(_skill())
    state = asyncio.run(solver(_make_state(), None))

    checker = step_check("nonexistent", lambda _r: True)
    with pytest.raises(KeyError) as exc:
        asyncio.run(checker(state, Target("x")))
    assert "nonexistent" in str(exc.value)


class TestModelRouting:
    """Verify that `@lm` calls route through Inspect's model."""

    def _mock_model(self, responses: list[str]):
        from unittest.mock import AsyncMock  # noqa: PLC0415

        from inspect_ai.model import ModelOutput  # noqa: PLC0415

        call_log: list[list] = []
        idx = [0]

        async def _generate(messages, **_kw):
            call_log.append(list(messages))
            text = responses[idx[0] % len(responses)]
            idx[0] += 1
            return ModelOutput.from_content(model="mock", content=text)

        mock = AsyncMock()
        mock.generate = _generate
        mock.__str__ = lambda _: "mock/model"
        return mock, call_log

    def _run_with_mock(self, skill, mock_model, input_text="hello", entry=None):
        from unittest.mock import patch  # noqa: PLC0415

        kw = {"entry": entry} if entry else {}
        solver = skill_solver(skill, **kw)
        state = _make_state(input_text=input_text)
        with patch("tk.llmbda.inspect._get_model", return_value=mock_model):
            return asyncio.run(solver(state, None))

    def test_lm_step_routes_through_model(self):
        from tk.llmbda import lm  # noqa: PLC0415

        def fake_caller(*, messages, **kw):  # noqa: ARG001
            return "should not be called"

        @lm(fake_caller, system_prompt="You are helpful.")
        def my_step(ctx: SkillContext, call) -> StepResult:
            raw = call(messages=[{"role": "user", "content": ctx.entry}])
            return StepResult(value=raw)

        skill = Skill(name="s", steps=[Skill("lm_step", fn=my_step)])
        model, call_log = self._mock_model(["model says hi"])
        out = self._run_with_mock(skill, model, input_text="test input")

        assert out.output.completion == "model says hi"
        assert len(call_log) == 1
        msgs = call_log[0]
        assert msgs[0].content == "You are helpful."
        assert msgs[1].content == "test input"

    def test_system_prompt_preserved_through_rebind(self):
        from tk.llmbda import lm  # noqa: PLC0415

        def orig_caller(*, messages, **kw):  # noqa: ARG001
            return "original"

        @lm(orig_caller, system_prompt="Extract dates.")
        def extract(ctx: SkillContext, call) -> StepResult:
            raw = call(messages=[{"role": "user", "content": ctx.entry}])
            return StepResult(value=raw)

        skill = Skill(name="s", steps=[Skill("ext", fn=extract)])
        model, call_log = self._mock_model(["2025-01-01"])
        out = self._run_with_mock(skill, model, input_text="jan first")

        assert out.output.completion == "2025-01-01"
        assert call_log[0][0].content == "Extract dates."

    def test_lm_step_appends_messages(self):
        from tk.llmbda import lm  # noqa: PLC0415

        def fake(*, messages, **kw):  # noqa: ARG001
            return "unused"

        @lm(fake, system_prompt="You are helpful.")
        def my_step(ctx: SkillContext, call) -> StepResult:
            raw = call(messages=[{"role": "user", "content": ctx.entry}])
            return StepResult(value=raw)

        skill = Skill(name="s", steps=[Skill("lm_step", fn=my_step)])
        model, _ = self._mock_model(["model says hi"])
        out = self._run_with_mock(skill, model, input_text="test input")

        assert [m.content for m in out.messages] == [
            "You are helpful.",
            "test input",
            "model says hi",
        ]

    def test_non_lm_steps_unaffected_by_rebind(self):
        def pure_step(ctx: SkillContext) -> StepResult:
            return StepResult(value=ctx.entry.upper())

        skill = Skill(name="s", steps=[Skill("upper", fn=pure_step)])
        model, call_log = self._mock_model(["unused"])
        out = self._run_with_mock(skill, model, input_text="hello")

        assert out.output.completion == "HELLO"
        assert call_log == []

    def test_multi_step_mixed_lm_and_pure(self):
        from tk.llmbda import lm  # noqa: PLC0415

        def normalize(ctx: SkillContext) -> StepResult:
            return StepResult(value=ctx.entry.strip().lower())

        def fake(*, messages, **kw):  # noqa: ARG001
            return "unused"

        @lm(fake, system_prompt="Classify.")
        def classify(ctx: SkillContext, call) -> StepResult:
            raw = call(messages=[{"role": "user", "content": ctx.prev.value}])
            return StepResult(value=raw)

        skill = Skill(
            name="s",
            steps=[Skill("norm", fn=normalize), Skill("cls", fn=classify)],
        )
        model, call_log = self._mock_model(["billing"])
        out = self._run_with_mock(skill, model, input_text="  HELLO  ")

        trace = out.metadata[DEFAULT_TRACE_KEY]
        assert trace["norm"].value == "hello"
        assert trace["cls"].value == "billing"
        assert out.output.completion == "billing"
        assert len(call_log) == 1
        assert call_log[0][0].content == "Classify."
        assert call_log[0][1].content == "hello"

    def test_trace_still_in_metadata(self):
        from tk.llmbda import lm  # noqa: PLC0415

        def fake(*, messages, **kw):  # noqa: ARG001
            return "x"

        @lm(fake, system_prompt="P")
        def step(_ctx: SkillContext, call) -> StepResult:
            return StepResult(value=call(messages=[{"role": "user", "content": "q"}]))

        skill = Skill(name="s", steps=[Skill("a", fn=step)])
        model, _ = self._mock_model(["response"])
        out = self._run_with_mock(skill, model)

        assert DEFAULT_TRACE_KEY in out.metadata
        assert "a" in out.metadata[DEFAULT_TRACE_KEY]
        assert out.metadata[DEFAULT_TRACE_KEY]["a"].value == "response"

    def test_async_lm_step_routes_through_model(self):
        """Native async @lm steps route through Inspect without thread bridge."""
        from tk.llmbda import lm  # noqa: PLC0415

        async def fake_caller(*, messages, **kw):  # noqa: ARG001
            return "should not be called"

        @lm(fake_caller, system_prompt="Async helper.")
        async def my_step(ctx: SkillContext, call) -> StepResult:
            raw = await call(messages=[{"role": "user", "content": ctx.entry}])
            return StepResult(value=raw)

        skill = Skill(name="s", steps=[Skill("lm_step", fn=my_step)])
        model, call_log = self._mock_model(["async model says hi"])
        out = self._run_with_mock(skill, model, input_text="test input")

        assert out.output.completion == "async model says hi"
        assert len(call_log) == 1
        assert call_log[0][0].content == "Async helper."
        assert call_log[0][1].content == "test input"

    def test_multi_call_within_single_step(self):
        """A step that calls the model multiple times gets all logged."""
        from tk.llmbda import lm  # noqa: PLC0415

        def fake(*, messages, **kw):  # noqa: ARG001
            return "x"

        @lm(fake, system_prompt="Multi.")
        def multi_call(_ctx: SkillContext, call) -> StepResult:
            r1 = call(messages=[{"role": "user", "content": "first"}])
            r2 = call(messages=[{"role": "user", "content": "second"}])
            return StepResult(value=f"{r1}+{r2}")

        skill = Skill(name="s", steps=[Skill("m", fn=multi_call)])
        model, call_log = self._mock_model(["A", "B"])
        out = self._run_with_mock(skill, model)

        assert out.output.completion == "A+B"
        assert len(call_log) == 2
