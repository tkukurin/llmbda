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
    last,
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
    match = re.search(r"(\d+)\s*(?:min|minute)s?", text)
    if match:
        return StepResult(value=int(match.group(1)), meta={"reason": "matched_minutes"})
    return StepResult(value=None, meta={"reason": "no_minute_match"})


def parse_hours(ctx: SkillContext) -> StepResult:
    """Find an exact hour match like '1.5 hours' or '2 hrs' and convert to minutes."""
    text = ctx.entry.get("text", "").lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?", text)
    if match:
        return StepResult(
            value=int(float(match.group(1)) * 60), meta={"reason": "matched_hours"}
        )
    return StepResult(value=None, meta={"reason": "no_hour_match"})


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


def _llm_diagnose(ctx: SkillContext, steps: list[Skill], call: LMCaller) -> StepResult:
    """Run parser children, fall back to LLM when none produce a value."""
    inner = Skill(name="_parse", steps=steps)
    r = run_skill(inner, ctx.entry)
    for sr in r.values():
        if sr.value is not None:
            return sr
    user_msg = json.dumps(
        {
            "text": ctx.entry.get("text", ""),
            "prior_steps": _prior_steps_payload(r, steps),
        },
        indent=2,
    )
    try:
        raw_response = call(messages=[{"role": "user", "content": user_msg}])
        parsed = json.loads(strip_fences(raw_response))
        return StepResult(
            value=parsed.get("value"),
            meta={"diagnosis": parsed.get("diagnosis", ""), "llm_raw": raw_response},
        )
    except Exception as exc:  # noqa: BLE001
        return StepResult(
            value=None, meta={"reason": "llm_parse_error", "error": str(exc)}
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
    },
    {
        "id": "T2_hours_regex",
        "text": "Roast the chicken for 1.5 hours.",
        "expected_val": 90,
    },
    {
        "id": "T3_llm_semantic_math",
        "text": "Pop it in the oven for half an hour.",
        "expected_val": 30,
    },
    {
        "id": "T4_llm_visual_cue",
        "text": "Cook until the crust is golden brown and the cheese is bubbly.",
        "expected_val": None,
    },
]


def test_parse_minutes_found():
    ctx = SkillContext(entry={"text": "Bake for 30 mins."})
    result = parse_minutes(ctx)
    assert result.value == 30
    assert result.meta["reason"] == "matched_minutes"


def test_parse_minutes_missing():
    ctx = SkillContext(entry={"text": "Bake until golden."})
    result = parse_minutes(ctx)
    assert result.value is None
    assert result.meta["reason"] == "no_minute_match"


def test_parse_hours_integer():
    ctx = SkillContext(entry={"text": "Roast for 2 hours."})
    assert parse_hours(ctx).value == 120


def test_parse_hours_decimal():
    ctx = SkillContext(entry={"text": "Roast for 1.5 hours."})
    assert parse_hours(ctx).value == 90


def test_parse_hours_missing():
    ctx = SkillContext(entry={"text": "Bake for 5 minutes."})
    assert parse_hours(ctx).value is None


def test_llm_diagnose_parses_json():
    def fake(**_kw: object) -> str:
        return '{"value": 30, "diagnosis": "half an hour = 30 min"}'

    ctx = SkillContext(entry={"text": "half an hour"})
    result = _llm_diagnose(ctx, [], fake)
    assert result.value == 30
    assert result.meta["diagnosis"] == "half an hour = 30 min"


def test_llm_diagnose_strips_fences():
    def fake(**_kw: object) -> str:
        return '```json\n{"value": 15, "diagnosis": "quarter hour"}\n```'

    ctx = SkillContext(entry={"text": "a quarter of an hour"})
    assert _llm_diagnose(ctx, [], fake).value == 15


def test_llm_diagnose_null_value():
    def fake(**_kw: object) -> str:
        return '{"value": null, "diagnosis": "no time mentioned"}'

    ctx = SkillContext(entry={"text": "cook until bubbly"})
    result = _llm_diagnose(ctx, [], fake)
    assert result.value is None
    assert result.meta["diagnosis"] == "no time mentioned"


def test_llm_diagnose_parse_error():
    def fake(**_kw: object) -> str:
        return "not JSON at all"

    ctx = SkillContext(entry={"text": "..."})
    result = _llm_diagnose(ctx, [], fake)
    assert result.value is None
    assert result.meta["reason"] == "llm_parse_error"


def test_prior_steps_payload_and_system_prompt():
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
    assert payload["prior_steps"][0]["value"] is None
    assert payload["prior_steps"][1]["value"] is None
    assert payload["prior_steps"][0]["metadata"] == {"reason": "no_minute_match"}
    assert payload["prior_steps"][1]["metadata"] == {"reason": "no_hour_match"}
    assert captured["system"] == LLM_SYSTEM_PROMPT


def test_orchestrator_short_circuits_on_resolved_parser():
    called = []

    def spy(*, messages: list[dict[str, str]], **_kw: object) -> str:  # noqa: ARG001
        called.append(True)
        return '{"value": 99, "diagnosis": "should not run"}'

    skill = _make_skill(spy)
    trace = run_skill(skill, {"text": "Bake for 45 mins."})
    assert last(trace).value == 45
    assert called == []


@pytest.mark.e2e
@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda x: x["id"])
def test_recipe_pipeline(tc, openai_caller):
    skill = _make_skill(openai_caller)
    trace = run_skill(skill, {"text": tc["text"]})
    assert last(trace).value == tc["expected_val"]
