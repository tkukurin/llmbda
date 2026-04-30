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
