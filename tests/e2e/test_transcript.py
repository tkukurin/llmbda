import json
import os
import re
import urllib.request
from typing import Any

import pytest

from tk.llmbda import Skill, StepContext, StepResult, run_skill

pytestmark = pytest.mark.e2e


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
            terminal=True,
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

Your task: Extract the single most important action item and its owner from the transcript.

Return ONLY a JSON object with exactly two keys:
{
  "task": "<a brief description of the task, or null>",
  "owner": "<the name of the person assigned, or null>"
}

Rules:
- If a person is explicitly asked to do something and agrees, or is told to do something, record it.
- If there is no clear action item, return null for both.
"""


def llm_extract_action(ctx: StepContext) -> StepResult:
    """Use an LLM to semantically infer tasks from conversational transcripts."""
    print("[Trace] Running llm_extract_action (Fallback Triggered)")
    if ctx.caller is None:
        print("  -> Failed: No caller provided.")
        return StepResult(
            value=None, metadata={"reason": "no_caller_available"}, terminal=True
        )

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
        cleaned = re.sub(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
            r"\1",
            raw_response.strip(),
            flags=re.DOTALL,
        )
        parsed = json.loads(cleaned)

        return StepResult(
            value=parsed if parsed.get("task") else None,
            metadata={"llm_raw": raw_response},
            terminal=True,
        )
    except Exception as exc:
        print(f"  -> LLM Error: {exc}")
        return StepResult(
            value=None,
            metadata={"reason": "llm_parse_error", "error": str(exc)},
            terminal=True,
        )


extract_action_item = Skill(
    name="extract_action_item",
    system_prompt=SYSTEM_PROMPT,
    steps=[parse_explicit_todo, llm_extract_action],
)


def openai_caller(**kwargs: Any) -> str:
    """Minimal synchronous caller for OpenAI's Chat API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")

    payload = json.dumps({"model": "gpt-4o-mini", "temperature": 0.0, **kwargs}).encode(
        "utf-8"
    )
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        return res["choices"][0]["message"]["content"]


TEST_CASES = [
    {
        "id": "T1_explicit_todo",
        "transcript": "Okay, so before we end. TODO: update the database schema @Sarah.",
        "expected_val": {"task": "update the database schema", "owner": "Sarah"},
        "expected_resolver": "parse_explicit_todo",
    },
    {
        "id": "T2_conversational_task",
        "transcript": "Alright, Bob, can you make sure to email the client by tomorrow morning?",
        "expected_val": {
            "task": "email the client by tomorrow morning",
            "owner": "Bob",
        },
        "expected_resolver": "llm_extract_action",
    },
    {
        "id": "T3_no_action_item",
        "transcript": "Great meeting everyone, I think we made a lot of progress today. See you all next week.",
        "expected_val": None,
        "expected_resolver": "llm_extract_action",
    },
]


@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda x: x["id"])
def test_transcript_pipeline(tc):
    print(f"\nTesting Case: {tc['id']}")

    caller = openai_caller

    result = run_skill(
        extract_action_item, {"transcript": tc["transcript"]}, caller=caller
    )
    print(
        f"\n[Final Output]\n{json.dumps({'value': result.value, 'metadata': result.metadata}, indent=2)}"
    )

    assert result.value == tc["expected_val"]
    assert result.metadata["resolved_by"] == tc["expected_resolver"]
