"""End-to-end multi-step pipeline test for llmbda — transcript action items."""

from __future__ import annotations

import json
import re

import pytest

from tk.llmbda import (
    LMCaller,
    Skill,
    Step,
    StepContext,
    StepResult,
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


def parse_explicit_todo(ctx: StepContext) -> StepResult:
    """Find an explicitly tagged action item of the form 'TODO: <task> @<owner>'."""
    text = ctx.entry.get("transcript", "")
    print(f"\n[Trace] Running parse_explicit_todo on '{text}'")
    match = re.search(r"TODO:\s*(.+?)\s+@(\w+)", text, re.IGNORECASE)
    if match:
        task, owner = match.groups()
        print(f"  -> Success: {task} assigned to {owner}")
        return StepResult(
            value={"task": task.strip(), "owner": owner.strip()},
            metadata={"reason": "explicit_todo_match"},
        )
    print("  -> Failed: No explicit TODO tag found.")
    return StepResult(
        value=None, metadata={"reason": "no_explicit_todo"}, resolved=False,
    )


def _prior_steps_payload(ctx: StepContext) -> list[dict[str, object]]:
    """Serialise prior steps with their intent and outcome for LLM context."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "value": ctx.prior[s.name].value,
            "metadata": ctx.prior[s.name].metadata,
        }
        for s in ctx.steps
        if s.name in ctx.prior
    ]


def _llm_extract_action(ctx: StepContext, call: LMCaller) -> StepResult:
    """Infer the most important action item and owner from a transcript."""
    print("[Trace] Running llm_extract_action (Fallback Triggered)")
    user_msg = json.dumps(
        {
            "transcript": ctx.entry.get("transcript", ""),
            "prior_steps": _prior_steps_payload(ctx),
        },
        indent=2,
    )
    print(f"  -> Prompting LLM with prior context:\n{user_msg}")

    try:
        raw_response = call(
            messages=[{"role": "user", "content": user_msg}],
        )
        print(f"  -> LLM Raw Response: {raw_response}")
        parsed = json.loads(strip_fences(raw_response))
        return StepResult(
            value=parsed if parsed.get("task") else None,
            metadata={"llm_raw": raw_response},
        )
    except Exception as exc:  # noqa: BLE001  -- step must survive any caller/parse failure
        print(f"  -> LLM Error: {exc}")
        return StepResult(
            value=None,
            metadata={"reason": "llm_parse_error", "error": str(exc)},
        )


def _make_skill(caller: LMCaller) -> Skill:
    return Skill(
        name="extract_action_item",
        steps=[
            Step("λ::todo", parse_explicit_todo),
            Step(
                "ψ::action",
                lm(caller, system_prompt=LLM_SYSTEM_PROMPT)(_llm_extract_action),
            ),
        ],
    )


TEST_CASES = [
    {
        "id": "T1_explicit_todo",
        "transcript": (
            "Okay, so before we end."
            " TODO: update the database schema @Sarah."
        ),
        "expected_val": {"task": "update the database schema", "owner": "Sarah"},
        "expected_resolver": "λ::todo",
    },
    {
        "id": "T2_conversational_task",
        "transcript": (
            "Alright, Bob, can you make sure to email"
            " the client by tomorrow morning?"
        ),
        "expected_val": {
            "task": "email the client by tomorrow morning",
            "owner": "Bob",
        },
        "expected_resolver": "ψ::action",
    },
    {
        "id": "T3_no_action_item",
        "transcript": (
            "Great meeting everyone, I think we made a lot"
            " of progress today. See you all next week."
        ),
        "expected_val": None,
        "expected_resolver": "ψ::action",
    },
]


def test_parse_explicit_todo_found():
    ctx = StepContext(
        entry={"transcript": "Before we end. TODO: ship the release @Alice."},
    )
    result = parse_explicit_todo(ctx)
    assert result.value == {"task": "ship the release", "owner": "Alice"}
    assert result.resolved is True
    assert result.metadata["reason"] == "explicit_todo_match"


def test_parse_explicit_todo_missing():
    ctx = StepContext(
        entry={"transcript": "Great work team, see you next week."},
    )
    result = parse_explicit_todo(ctx)
    assert result.value is None
    assert result.resolved is False
    assert result.metadata["reason"] == "no_explicit_todo"


def test_llm_extract_action_parses():
    def fake(**_kw: object) -> str:
        return '{"task": "email the client", "owner": "Bob"}'

    ctx = StepContext(entry={"transcript": "Bob, email the client tomorrow."})
    result = _llm_extract_action(ctx, fake)
    assert result.value == {"task": "email the client", "owner": "Bob"}


def test_llm_extract_action_strips_fences():
    def fake(**_kw: object) -> str:
        return '```json\n{"task": "review PR", "owner": "Carol"}\n```'

    ctx = StepContext(entry={"transcript": "Carol, PR review please."})
    result = _llm_extract_action(ctx, fake)
    assert result.value == {"task": "review PR", "owner": "Carol"}


def test_llm_extract_action_null_when_no_task():
    def fake(**_kw: object) -> str:
        return '{"task": null, "owner": null}'

    ctx = StepContext(entry={"transcript": "Nice catching up."})
    result = _llm_extract_action(ctx, fake)
    assert result.value is None


def test_llm_extract_action_parse_error():
    def fake(**_kw: object) -> str:
        return "not JSON"

    ctx = StepContext(entry={"transcript": "..."})
    result = _llm_extract_action(ctx, fake)
    assert result.value is None
    assert result.metadata["reason"] == "llm_parse_error"


def test_prior_steps_payload_and_system_prompt():
    """llm_extract_action receives prior step descriptions and its own system prompt."""
    captured: dict[str, str] = {}

    def spy(*, messages: list[dict[str, str]], **_kw: object) -> str:
        captured["system"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return '{"task": null, "owner": null}'

    skill = _make_skill(spy)
    run_skill(skill, {"transcript": "Great meeting everyone."})
    payload = json.loads(captured["user"])
    names = [s["name"] for s in payload["prior_steps"]]
    assert names == ["λ::todo"]
    assert payload["prior_steps"][0]["description"] == (
        "Find an explicitly tagged action item of the form 'TODO: <task> @<owner>'."
    )
    assert payload["prior_steps"][0]["value"] is None
    assert captured["system"] == LLM_SYSTEM_PROMPT


@pytest.mark.e2e
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda x: x["id"])
def test_transcript_pipeline(tc, openai_caller):
    print(f"\nTesting Case: {tc['id']}")

    skill = _make_skill(openai_caller)
    result = run_skill(skill, {"transcript": tc["transcript"]})
    output = {
        "skill": result.skill,
        "resolved_by": result.resolved_by,
        "value": result.value,
        "metadata": result.metadata,
    }
    print(f"\n[Final Output]\n{json.dumps(output, indent=2)}")

    assert result.value == tc["expected_val"]
    assert result.resolved_by == tc["expected_resolver"]
