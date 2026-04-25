"""End-to-end multi-step pipeline test for llmbda."""

from __future__ import annotations

import json
import re

import pytest

from tk.llmbda import Skill, Step, StepContext, StepResult, run_skill, strip_fences


def _noop(**_kw: object) -> None:
    return None


PARSE_MINUTES_PROMPT = "Find an exact minute match like '45 mins' or '30 minutes'."
PARSE_HOURS_PROMPT = (
    "Find an exact hour match like '1.5 hours' or '2 hrs' and convert to minutes."
)

LLM_SYSTEM_PROMPT = """\
You are a culinary data extractor.
You receive a JSON object with:
- "text": A messy snippet from a recipe blog.
- "prior_steps": The earlier parsers that ran, each with its own intent
  (system_prompt) and outcome (metadata).

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


def parse_minutes(ctx: StepContext) -> StepResult:
    """Try to find an exact minute match (e.g., '45 mins')."""
    text = ctx.entry.get("text", "").lower()
    print(f"\n[Trace] Running Step 1: parse_minutes on '{text}'")
    match = re.search(r"(\d+)\s*(?:min|minute)s?", text)
    if match:
        mins = int(match.group(1))
        print(f"  -> Success: Found {mins} minutes.")
        return StepResult(value=mins, metadata={"reason": "matched_minutes"})
    print("  -> Failed: No minute regex match.")
    return StepResult(
        value=None, metadata={"reason": "no_minute_match"}, resolved=False,
    )


def parse_hours(ctx: StepContext) -> StepResult:
    """Try to find an exact hour match and convert to minutes (e.g., '1.5 hours')."""
    text = ctx.entry.get("text", "").lower()
    print(f"[Trace] Running Step 2: parse_hours on '{text}'")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?", text)
    if match:
        mins = int(float(match.group(1)) * 60)
        print(f"  -> Success: Found hours, converted to {mins} minutes.")
        return StepResult(value=mins, metadata={"reason": "matched_hours"})
    print("  -> Failed: No hour regex match.")
    return StepResult(value=None, metadata={"reason": "no_hour_match"}, resolved=False)


def _prior_steps_payload(ctx: StepContext) -> list[dict[str, object]]:
    """Serialise prior steps with their intent and outcome for LLM context."""
    return [
        {
            "name": s.name,
            "system_prompt": s.system_prompt,
            "metadata": ctx.prior[s.name].metadata,
        }
        for s in ctx.steps
        if s.name in ctx.prior
    ]


def llm_diagnose(ctx: StepContext) -> StepResult:
    """Use an LLM to semantically parse the time or diagnose the failure."""
    print("[Trace] Running Step 3: llm_diagnose (Fallback Triggered)")
    user_msg = json.dumps(
        {"text": ctx.entry.get("text", ""), "prior_steps": _prior_steps_payload(ctx)},
        indent=2,
    )
    print(f"  -> Prompting LLM with prior context:\n{user_msg}")

    try:
        raw_response = ctx.caller(
            messages=[{"role": "user", "content": user_msg}],
        )
        print(f"  -> LLM Raw Response: {raw_response}")
        parsed = json.loads(strip_fences(raw_response))
        return StepResult(
            value=parsed.get("value"),
            metadata={
                "diagnosis": parsed.get("diagnosis", ""),
                "llm_raw": raw_response,
            },
        )
    except Exception as exc:  # noqa: BLE001  -- step must survive any caller/parse failure
        print(f"  -> LLM Error: {exc}")
        return StepResult(
            value=None,
            metadata={"reason": "llm_parse_error", "error": str(exc)},
        )


extract_cook_time = Skill(
    name="extract_cook_time",
    steps=[
        Step("parse_minutes", parse_minutes, system_prompt=PARSE_MINUTES_PROMPT),
        Step("parse_hours", parse_hours, system_prompt=PARSE_HOURS_PROMPT),
        Step("llm_diagnose", llm_diagnose, system_prompt=LLM_SYSTEM_PROMPT),
    ],
)


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


def test_parse_minutes_found():
    ctx = StepContext(entry={"text": "Bake for 30 mins."}, caller=_noop)
    result = parse_minutes(ctx)
    assert result.value == 30
    assert result.resolved is True
    assert result.metadata["reason"] == "matched_minutes"


def test_parse_minutes_missing():
    ctx = StepContext(entry={"text": "Bake until golden."}, caller=_noop)
    result = parse_minutes(ctx)
    assert result.value is None
    assert result.resolved is False
    assert result.metadata["reason"] == "no_minute_match"


def test_parse_hours_integer():
    ctx = StepContext(entry={"text": "Roast for 2 hours."}, caller=_noop)
    result = parse_hours(ctx)
    assert result.value == 120
    assert result.resolved is True


def test_parse_hours_decimal():
    ctx = StepContext(entry={"text": "Roast for 1.5 hours."}, caller=_noop)
    result = parse_hours(ctx)
    assert result.value == 90
    assert result.resolved is True


def test_parse_hours_missing():
    ctx = StepContext(entry={"text": "Bake for 5 minutes."}, caller=_noop)
    result = parse_hours(ctx)
    assert result.value is None
    assert result.resolved is False


def test_llm_diagnose_parses_json():
    def fake(**_kw):
        return '{"value": 30, "diagnosis": "half an hour = 30 min"}'

    ctx = StepContext(entry={"text": "half an hour"}, caller=fake)
    result = llm_diagnose(ctx)
    assert result.value == 30
    assert result.metadata["diagnosis"] == "half an hour = 30 min"


def test_llm_diagnose_strips_fences():
    def fake(**_kw):
        return '```json\n{"value": 15, "diagnosis": "quarter hour"}\n```'

    ctx = StepContext(entry={"text": "a quarter of an hour"}, caller=fake)
    result = llm_diagnose(ctx)
    assert result.value == 15


def test_llm_diagnose_null_value():
    def fake(**_kw):
        return '{"value": null, "diagnosis": "no time mentioned"}'

    ctx = StepContext(entry={"text": "cook until bubbly"}, caller=fake)
    result = llm_diagnose(ctx)
    assert result.value is None
    assert result.metadata["diagnosis"] == "no time mentioned"


def test_llm_diagnose_parse_error():
    def fake(**_kw):
        return "not JSON at all"

    ctx = StepContext(entry={"text": "..."}, caller=fake)
    result = llm_diagnose(ctx)
    assert result.value is None
    assert result.metadata["reason"] == "llm_parse_error"


def test_prior_steps_payload_includes_system_prompts():
    """llm_diagnose receives earlier steps' intents, not just their metadata."""
    captured: dict[str, str] = {}

    def fake(**kwargs):
        captured["user"] = kwargs["messages"][1]["content"]
        captured["system"] = kwargs["messages"][0]["content"]
        return '{"value": null, "diagnosis": "nothing"}'

    run_skill(extract_cook_time, {"text": "cook until bubbly"}, caller=fake)
    payload = json.loads(captured["user"])
    names = [s["name"] for s in payload["prior_steps"]]
    assert names == ["parse_minutes", "parse_hours"]
    assert payload["prior_steps"][0]["system_prompt"] == PARSE_MINUTES_PROMPT
    assert payload["prior_steps"][1]["system_prompt"] == PARSE_HOURS_PROMPT
    assert captured["system"] == LLM_SYSTEM_PROMPT


@pytest.mark.e2e
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda x: x["id"])
def test_recipe_pipeline(tc, openai_caller):
    print(f"\nTesting Case: {tc['id']}")

    result = run_skill(extract_cook_time, {"text": tc["text"]}, caller=openai_caller)
    output = {
        "skill": result.skill,
        "resolved_by": result.resolved_by,
        "value": result.value,
        "metadata": result.metadata,
    }
    print(f"\n[Final Output]\n{json.dumps(output, indent=2)}")

    assert result.value == tc["expected_val"]
    assert result.resolved_by == tc["expected_resolver"]
