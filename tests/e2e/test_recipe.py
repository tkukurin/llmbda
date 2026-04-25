"""End-to-end multi-step pipeline test for llmbda."""

import json
import os
import re
import urllib.request
from typing import Any

import pytest

from tk.llmbda import Skill, StepContext, StepResult, run_skill

pytestmark = pytest.mark.e2e


def parse_minutes(ctx: StepContext) -> StepResult:
    """Try to find an exact minute match (e.g., '45 mins')."""
    text = ctx.entry.get("text", "").lower()
    print(f"\n[Trace] Running Step 1: parse_minutes on '{text}'")
    match = re.search(r"(\d+)\s*(?:min|minute)s?", text)
    if match:
        mins = int(match.group(1))
        print(f"  -> Success: Found {mins} minutes.")
        return StepResult(
            value=mins, metadata={"reason": "matched_minutes"}, terminal=True
        )
    print("  -> Failed: No minute regex match.")
    return StepResult(
        value=None, metadata={"reason": "no_minute_match"}, terminal=False
    )


def parse_hours(ctx: StepContext) -> StepResult:
    """Try to find an exact hour match and convert to minutes (e.g., '1.5 hours')."""
    text = ctx.entry.get("text", "").lower()
    print(f"[Trace] Running Step 2: parse_hours on '{text}'")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?", text)
    if match:
        mins = int(float(match.group(1)) * 60)
        print(f"  -> Success: Found hours, converted to {mins} minutes.")
        return StepResult(
            value=mins, metadata={"reason": "matched_hours"}, terminal=True
        )
    print("  -> Failed: No hour regex match.")
    return StepResult(value=None, metadata={"reason": "no_hour_match"}, terminal=False)


SYSTEM_PROMPT = """\
You are a culinary data extractor.
You receive a JSON object with:
- "text": A messy snippet from a recipe blog.
- "prior_steps": The results of our deterministic regex parsers which failed.

Your task: Extract the cooking time in minutes, or explain why it's missing.

Return ONLY a JSON object with exactly two keys:
{
  "value": <int minutes if found/inferred, else null>,
  "diagnosis": "<one-sentence explanation of why regex failed, or why time is absent>"
}

Rules:
- If the text says "half an hour", value is 30.
- If it relies on visual cues ("until golden brown") with no time, value is null.
"""


def llm_diagnose(ctx: StepContext) -> StepResult:
    """Use an LLM to semantically parse the time or diagnose the failure."""
    print("[Trace] Running Step 3: llm_diagnose (Fallback Triggered)")
    if ctx.caller is None:
        print("  -> Failed: No caller provided.")
        return StepResult(
            value=None, metadata={"reason": "no_caller_available"}, terminal=True
        )

    prior_meta = {k: v.metadata for k, v in ctx.prior.items()}
    user_msg = json.dumps(
        {"text": ctx.entry.get("text", ""), "prior_steps": prior_meta}, indent=2
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
            value=parsed.get("value"),
            metadata={
                "diagnosis": parsed.get("diagnosis", ""),
                "llm_raw": raw_response,
            },
            terminal=True,
        )
    except Exception as exc:
        print(f"  -> LLM Error: {exc}")
        return StepResult(
            value=None,
            metadata={"reason": "llm_parse_error", "error": str(exc)},
            terminal=True,
        )


extract_cook_time = Skill(
    name="extract_cook_time",
    system_prompt=SYSTEM_PROMPT,
    steps=[parse_minutes, parse_hours, llm_diagnose],
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
        "id": "T1_minutes_regex",
        "text": "Bake the cookies for 45 mins at 350 degrees.",
        "expected_val": 45,
        "expected_resolver": "parse_minutes",
    },
    {
        "id": "T2_hours_regex",
        "text": "Roast the chicken for 1.5 hours.",
        "expected_val": 90,
        "expected_resolver": "parse_hours",
    },
    {
        "id": "T3_llm_semantic_math",
        "text": "Pop it in the oven for half an hour.",
        "expected_val": 30,
        "expected_resolver": "llm_diagnose",
    },
    {
        "id": "T4_llm_visual_cue",
        "text": "Cook until the crust is golden brown and the cheese is bubbly.",
        "expected_val": None,
        "expected_resolver": "llm_diagnose",
    },
]


@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda x: x["id"])
def test_recipe_pipeline(tc):
    print(f"\nTesting Case: {tc['id']}")

    caller = openai_caller

    result = run_skill(extract_cook_time, {"text": tc["text"]}, caller=caller)
    print(
        f"\n[Final Output]\n{json.dumps({'value': result.value, 'metadata': result.metadata}, indent=2)}"
    )

    assert result.value == tc["expected_val"]
    assert result.metadata["resolved_by"] == tc["expected_resolver"]
