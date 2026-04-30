"""End-to-end multi-step pipeline test for llmbda — transcript action items."""

from __future__ import annotations

import json
import re

import pytest

from tk.llmbda import (
    LMCaller,
    Skill,
    SkillContext,
    StepResult,
    last,
    lm,
    run_skill,
    strip_fences,
)

LLM_SYSTEM_PROMPT = """\
You are an AI meeting assistant.
You receive a JSON object with:
- "transcript": A snippet of dialogue from a meeting.
- "prior_steps": The earlier parsers that ran, each with its own intent
  (description), value, and metadata.

Your task: Extract the single most important action item
and its owner from the transcript.

Return ONLY a JSON object with exactly two keys:
{
  "task": "<a brief description of the task, or null>",
  "owner": "<the name of the person assigned, or null>"
}

Rules:
- If a person is explicitly asked to do something and agrees,
  or is told to do something, record it.
- If there is no clear action item, return null for both.
"""


def parse_explicit_todo(ctx: SkillContext) -> StepResult:
    """Find an explicitly tagged action item of the form 'TODO: <task> @<owner>'."""
    text = ctx.entry.get("transcript", "")
    match = re.search(r"TODO:\s*(.+?)\s+@(\w+)", text, re.IGNORECASE)
    if match:
        task, owner = match.groups()
        return StepResult(
            value={"task": task.strip(), "owner": owner.strip()},
            meta={"reason": "explicit_todo_match"},
        )
    return StepResult(value=None, meta={"reason": "no_explicit_todo"})


def _prior_steps_payload(
    trace: dict[str, StepResult],
    skills: list[Skill],
) -> list[dict[str, object]]:
    return [
        {
            "name": s.name,
            "description": s.description,
            "value": trace[s.name].value,
            "metadata": trace[s.name].meta,
        }
        for s in skills
        if s.name in trace
    ]


def _llm_extract_action(
    ctx: SkillContext,
    steps: list[Skill],
    call: LMCaller,
) -> StepResult:
    """Run parser children, then infer the action item via LLM on fallback."""
    inner = Skill(name="_parse", steps=steps)
    r = run_skill(inner, ctx.entry)
    for sr in r.values():
        if sr.value is not None:
            return sr
    user_msg = json.dumps(
        {
            "transcript": ctx.entry.get("transcript", ""),
            "prior_steps": _prior_steps_payload(r, steps),
        },
        indent=2,
    )
    try:
        raw_response = call(messages=[{"role": "user", "content": user_msg}])
        parsed = json.loads(strip_fences(raw_response))
        return StepResult(
            value=parsed if parsed.get("task") else None,
            meta={"llm_raw": raw_response},
        )
    except Exception as exc:  # noqa: BLE001
        return StepResult(
            value=None, meta={"reason": "llm_parse_error", "error": str(exc)}
        )


def _make_skill(caller: LMCaller) -> Skill:
    return Skill(
        name="extract_action_item",
        fn=lm(caller, system_prompt=LLM_SYSTEM_PROMPT)(_llm_extract_action),
        steps=[Skill("λ::todo", fn=parse_explicit_todo)],
    )


TEST_CASES = [
    {
        "id": "T1_explicit_todo",
        "transcript": (
            "Okay, so before we end. TODO: update the database schema @Sarah."
        ),
        "expected_val": {"task": "update the database schema", "owner": "Sarah"},
    },
    {
        "id": "T2_conversational_task",
        "transcript": (
            "Alright, Bob, can you make sure to email the client by tomorrow morning?"
        ),
        "expected_val": {
            "task": "email the client by tomorrow morning",
            "owner": "Bob",
        },
    },
    {
        "id": "T3_no_action_item",
        "transcript": (
            "Great meeting everyone, I think we made a lot"
            " of progress today. See you all next week."
        ),
        "expected_val": None,
    },
]


def test_parse_explicit_todo_found():
    ctx = SkillContext(
        entry={"transcript": "Before we end. TODO: ship the release @Alice."}
    )
    result = parse_explicit_todo(ctx)
    assert result.value == {"task": "ship the release", "owner": "Alice"}
    assert result.meta["reason"] == "explicit_todo_match"


def test_parse_explicit_todo_missing():
    ctx = SkillContext(entry={"transcript": "Great work team, see you next week."})
    result = parse_explicit_todo(ctx)
    assert result.value is None
    assert result.meta["reason"] == "no_explicit_todo"


def test_llm_extract_action_parses():
    def fake(**_kw: object) -> str:
        return '{"task": "email the client", "owner": "Bob"}'

    ctx = SkillContext(entry={"transcript": "Bob, email the client tomorrow."})
    result = _llm_extract_action(ctx, [], fake)
    assert result.value == {"task": "email the client", "owner": "Bob"}


def test_llm_extract_action_strips_fences():
    def fake(**_kw: object) -> str:
        return '```json\n{"task": "review PR", "owner": "Carol"}\n```'

    ctx = SkillContext(entry={"transcript": "Carol, PR review please."})
    assert _llm_extract_action(ctx, [], fake).value == {
        "task": "review PR",
        "owner": "Carol",
    }


def test_llm_extract_action_null_when_no_task():
    def fake(**_kw: object) -> str:
        return '{"task": null, "owner": null}'

    ctx = SkillContext(entry={"transcript": "Nice catching up."})
    assert _llm_extract_action(ctx, [], fake).value is None


def test_llm_extract_action_parse_error():
    def fake(**_kw: object) -> str:
        return "not JSON"

    ctx = SkillContext(entry={"transcript": "..."})
    result = _llm_extract_action(ctx, [], fake)
    assert result.value is None
    assert result.meta["reason"] == "llm_parse_error"


def test_prior_steps_payload_and_system_prompt():
    captured: dict[str, str] = {}

    def spy(*, messages: list[dict[str, str]], **_kw: object) -> str:
        captured["system"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return '{"task": null, "owner": null}'

    skill = _make_skill(spy)
    run_skill(skill, {"transcript": "Great meeting everyone."})
    payload = json.loads(captured["user"])
    assert payload["prior_steps"][0]["value"] is None
    assert captured["system"] == LLM_SYSTEM_PROMPT


@pytest.mark.e2e
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda x: x["id"])
def test_transcript_pipeline(tc, openai_caller):
    skill = _make_skill(openai_caller)
    trace = run_skill(skill, {"transcript": tc["transcript"]})
    assert last(trace).value == tc["expected_val"]
