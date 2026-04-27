"""End-to-end multi-step pipeline test for llmbda."""

from __future__ import annotations

import json
import re

import pytest

from tk.llmbda import (
    LMCaller,
    Skill,
    SkillContext,
    StepResult,
    lm,
    run_skill,
    strip_fences,
)

LLM_SYSTEM_PROMPT = """\
You are a culinary data extractor.
You receive a JSON object with:
- "text": A messy snippet from a recipe blog.
- "prior_steps": The earlier parsers that ran, each with its own intent
  (description), value, and metadata.

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


def parse_minutes(ctx: SkillContext) -> StepResult:
    """Find an exact minute match like '45 mins' or '30 minutes'."""
    text = ctx.entry.get("text", "").lower()
    print(f"\n[Trace] Running Step 1: parse_minutes on '{text}'")
    match = re.search(r"(\d+)\s*(?:min|minute)s?", text)
    if match:
        mins = int(match.group(1))
        print(f"  -> Success: Found {mins} minutes.")
        return StepResult(
            value=mins,
            metadata={"reason": "matched_minutes"},
            resolved=True,
        )
    print("  -> Failed: No minute regex match.")
    return StepResult(
        value=None,
        metadata={"reason": "no_minute_match"},
    )


def parse_hours(ctx: SkillContext) -> StepResult:
    """Find an exact hour match like '1.5 hours' or '2 hrs' and convert to minutes."""
    text = ctx.entry.get("text", "").lower()
    print(f"[Trace] Running Step 2: parse_hours on '{text}'")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?", text)
    if match:
        mins = int(float(match.group(1)) * 60)
        print(f"  -> Success: Found hours, converted to {mins} minutes.")
        return StepResult(
            value=mins,
            metadata={"reason": "matched_hours"},
            resolved=True,
        )
    print("  -> Failed: No hour regex match.")
    return StepResult(value=None, metadata={"reason": "no_hour_match"})


def _prior_steps_payload(
    trace: dict[str, StepResult],
    skills: list[Skill],
) -> list[dict[str, object]]:
    """Serialise prior steps with their intent, value, and metadata."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "value": trace[s.name].value,
            "metadata": trace[s.name].metadata,
        }
        for s in skills
        if s.name in trace
    ]


def _llm_diagnose(ctx: SkillContext, steps: list[Skill], call: LMCaller) -> StepResult:
    """Run parser children, fall back to LLM when none resolve."""
    inner = Skill(name="_parse", steps=steps)
    r = run_skill(inner, ctx.entry)
    if any(sr.resolved for sr in r.trace.values()):
        return StepResult(value=r.value, metadata=r.metadata, resolved_by=r.resolved_by)
    print("[Trace] Running Step 3: llm_diagnose (Fallback Triggered)")
    user_msg = json.dumps(
        {
            "text": ctx.entry.get("text", ""),
            "prior_steps": _prior_steps_payload(r.trace, steps),
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


def _make_skill(caller: LMCaller) -> Skill:
    return Skill(
        name="extract_cook_time",
        fn=lm(caller, system_prompt=LLM_SYSTEM_PROMPT)(_llm_diagnose),
        steps=[
            Skill("λ::minutes", fn=parse_minutes),
            Skill("λ::hours", fn=parse_hours),
        ],
    )


TEST_CASES = [
    {
        "id": "T1_minutes_regex",
        "text": "Bake the cookies for 45 mins at 350 degrees.",
        "expected_val": 45,
        "expected_resolver": ("extract_cook_time", "λ::minutes"),
    },
    {
        "id": "T2_hours_regex",
        "text": "Roast the chicken for 1.5 hours.",
        "expected_val": 90,
        "expected_resolver": ("extract_cook_time", "λ::hours"),
    },
    {
        "id": "T3_llm_semantic_math",
        "text": "Pop it in the oven for half an hour.",
        "expected_val": 30,
        "expected_resolver": ("extract_cook_time",),
    },
    {
        "id": "T4_llm_visual_cue",
        "text": "Cook until the crust is golden brown and the cheese is bubbly.",
        "expected_val": None,
        "expected_resolver": ("extract_cook_time",),
    },
]


def test_parse_minutes_found():
    ctx = SkillContext(entry={"text": "Bake for 30 mins."})
    result = parse_minutes(ctx)
    assert result.value == 30
    assert result.resolved is True
    assert result.metadata["reason"] == "matched_minutes"


def test_parse_minutes_missing():
    ctx = SkillContext(entry={"text": "Bake until golden."})
    result = parse_minutes(ctx)
    assert result.value is None
    assert result.resolved is False
    assert result.metadata["reason"] == "no_minute_match"


def test_parse_hours_integer():
    ctx = SkillContext(entry={"text": "Roast for 2 hours."})
    result = parse_hours(ctx)
    assert result.value == 120
    assert result.resolved is True


def test_parse_hours_decimal():
    ctx = SkillContext(entry={"text": "Roast for 1.5 hours."})
    result = parse_hours(ctx)
    assert result.value == 90
    assert result.resolved is True


def test_parse_hours_missing():
    ctx = SkillContext(entry={"text": "Bake for 5 minutes."})
    result = parse_hours(ctx)
    assert result.value is None
    assert result.resolved is False


def test_llm_diagnose_parses_json():
    def fake(**_kw: object) -> str:
        return '{"value": 30, "diagnosis": "half an hour = 30 min"}'

    ctx = SkillContext(entry={"text": "half an hour"})
    result = _llm_diagnose(ctx, [], fake)
    assert result.value == 30
    assert result.metadata["diagnosis"] == "half an hour = 30 min"


def test_llm_diagnose_strips_fences():
    def fake(**_kw: object) -> str:
        return '```json\n{"value": 15, "diagnosis": "quarter hour"}\n```'

    ctx = SkillContext(entry={"text": "a quarter of an hour"})
    result = _llm_diagnose(ctx, [], fake)
    assert result.value == 15


def test_llm_diagnose_null_value():
    def fake(**_kw: object) -> str:
        return '{"value": null, "diagnosis": "no time mentioned"}'

    ctx = SkillContext(entry={"text": "cook until bubbly"})
    result = _llm_diagnose(ctx, [], fake)
    assert result.value is None
    assert result.metadata["diagnosis"] == "no time mentioned"


def test_llm_diagnose_parse_error():
    def fake(**_kw: object) -> str:
        return "not JSON at all"

    ctx = SkillContext(entry={"text": "..."})
    result = _llm_diagnose(ctx, [], fake)
    assert result.value is None
    assert result.metadata["reason"] == "llm_parse_error"


def test_prior_steps_payload_and_system_prompt():
    """llm_diagnose receives prior step descriptions and its own system prompt."""
    captured: dict[str, str] = {}

    def spy(*, messages: list[dict[str, str]], **_kw: object) -> str:
        captured["system"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return '{"value": null, "diagnosis": "nothing"}'

    skill = _make_skill(spy)
    run_skill(skill, {"text": "cook until bubbly"})
    payload = json.loads(captured["user"])
    names = [s["name"] for s in payload["prior_steps"]]
    assert names == ["λ::minutes", "λ::hours"]
    assert payload["prior_steps"][0]["description"] == (
        "Find an exact minute match like '45 mins' or '30 minutes'."
    )
    assert payload["prior_steps"][1]["description"] == (
        "Find an exact hour match like '1.5 hours' or '2 hrs' and convert to minutes."
    )
    assert payload["prior_steps"][0]["value"] is None
    assert payload["prior_steps"][1]["value"] is None
    assert payload["prior_steps"][0]["metadata"] == {"reason": "no_minute_match"}
    assert payload["prior_steps"][1]["metadata"] == {"reason": "no_hour_match"}
    assert captured["system"] == LLM_SYSTEM_PROMPT


def test_orchestrator_short_circuits_on_resolved_parser():
    """When a parser resolves, the orchestrator returns without calling LLM."""
    called = []

    def spy(*, messages: list[dict[str, str]], **_kw: object) -> str:  # noqa: ARG001
        called.append(True)
        return '{"value": 99, "diagnosis": "should not run"}'

    skill = _make_skill(spy)
    result = run_skill(skill, {"text": "Bake for 45 mins."})
    assert result.value == 45
    assert called == []


@pytest.mark.e2e
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda x: x["id"])
def test_recipe_pipeline(tc, openai_caller):
    print(f"\nTesting Case: {tc['id']}")

    skill = _make_skill(openai_caller)
    result = run_skill(skill, {"text": tc["text"]})
    output = {
        "skill": result.skill,
        "resolved_by": result.resolved_by,
        "value": result.value,
        "metadata": result.metadata,
    }
    print(f"\n[Final Output]\n{json.dumps(output, indent=2)}")

    assert result.value == tc["expected_val"]
    assert result.resolved_by == tc["expected_resolver"]
