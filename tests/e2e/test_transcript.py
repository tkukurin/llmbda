import json
import re

import pytest

from tk.llmbda import Skill, Step, StepContext, StepResult, run_skill, strip_fences


def _noop(**_kw: object) -> None:
    return None


def parse_explicit_todo(ctx: StepContext) -> StepResult:
    """Extract an explicitly marked TODO and owner."""
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
        value=None, metadata={"reason": "no_explicit_todo"}, terminal=False
    )


SYSTEM_PROMPT = """\
You are an AI meeting assistant.
You receive a JSON object with:
- "transcript": A snippet of dialogue from a meeting.
- "prior_steps": Output from deterministic parsers that failed.

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


def llm_extract_action(ctx: StepContext) -> StepResult:
    """Use an LLM to semantically infer tasks from conversational transcripts."""
    print("[Trace] Running llm_extract_action (Fallback Triggered)")
    prior_meta = {k: v.metadata for k, v in ctx.prior.items()}
    user_msg = json.dumps(
        {"transcript": ctx.entry.get("transcript", ""), "prior_steps": prior_meta},
        indent=2,
    )
    print(f"  -> Prompting LLM with prior context:\n{user_msg}")

    try:
        raw_response = ctx.caller(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
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


extract_action_item = Skill(
    name="extract_action_item",
    system_prompt=SYSTEM_PROMPT,
    steps=[
        Step("parse_explicit_todo", parse_explicit_todo),
        Step("llm_extract_action", llm_extract_action),
    ],
)


TEST_CASES = [
    {
        "id": "T1_explicit_todo",
        "transcript": "Okay, so before we end. TODO: update the database schema @Sarah.",  # noqa: E501
        "expected_val": {"task": "update the database schema", "owner": "Sarah"},
        "expected_resolver": "parse_explicit_todo",
    },
    {
        "id": "T2_conversational_task",
        "transcript": "Alright, Bob, can you make sure to email the client by tomorrow morning?",  # noqa: E501
        "expected_val": {
            "task": "email the client by tomorrow morning",
            "owner": "Bob",
        },
        "expected_resolver": "llm_extract_action",
    },
    {
        "id": "T3_no_action_item",
        "transcript": "Great meeting everyone, I think we made a lot of progress today. See you all next week.",  # noqa: E501
        "expected_val": None,
        "expected_resolver": "llm_extract_action",
    },
]


def test_parse_explicit_todo_found():
    ctx = StepContext(
        entry={"transcript": "Before we end. TODO: ship the release @Alice."},
        caller=_noop,
    )
    result = parse_explicit_todo(ctx)
    assert result.value == {"task": "ship the release", "owner": "Alice"}
    assert result.terminal is True
    assert result.metadata["reason"] == "explicit_todo_match"


def test_parse_explicit_todo_missing():
    ctx = StepContext(
        entry={"transcript": "Great work team, see you next week."},
        caller=_noop,
    )
    result = parse_explicit_todo(ctx)
    assert result.value is None
    assert result.terminal is False
    assert result.metadata["reason"] == "no_explicit_todo"


def test_llm_extract_action_parses():
    def fake(**_kw):
        return '{"task": "email the client", "owner": "Bob"}'

    ctx = StepContext(
        entry={"transcript": "Bob, email the client tomorrow."}, caller=fake,
    )
    result = llm_extract_action(ctx)
    assert result.value == {"task": "email the client", "owner": "Bob"}


def test_llm_extract_action_strips_fences():
    def fake(**_kw):
        return '```json\n{"task": "review PR", "owner": "Carol"}\n```'

    ctx = StepContext(entry={"transcript": "Carol, PR review please."}, caller=fake)
    result = llm_extract_action(ctx)
    assert result.value == {"task": "review PR", "owner": "Carol"}


def test_llm_extract_action_null_when_no_task():
    def fake(**_kw):
        return '{"task": null, "owner": null}'

    ctx = StepContext(entry={"transcript": "Nice catching up."}, caller=fake)
    result = llm_extract_action(ctx)
    assert result.value is None


def test_llm_extract_action_parse_error():
    def fake(**_kw):
        return "not JSON"

    ctx = StepContext(entry={"transcript": "..."}, caller=fake)
    result = llm_extract_action(ctx)
    assert result.value is None
    assert result.metadata["reason"] == "llm_parse_error"


@pytest.mark.e2e
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda x: x["id"])
def test_transcript_pipeline(tc, openai_caller):
    print(f"\nTesting Case: {tc['id']}")

    result = run_skill(
        extract_action_item,
        {"transcript": tc["transcript"]},
        caller=openai_caller,
    )
    output = {
        "skill": result.skill,
        "resolved_by": result.resolved_by,
        "value": result.value,
        "metadata": result.metadata,
    }
    print(f"\n[Final Output]\n{json.dumps(output, indent=2)}")

    assert result.value == tc["expected_val"]
    assert result.resolved_by == tc["expected_resolver"]
