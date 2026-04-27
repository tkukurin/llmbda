# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=2.32.0",
#     "tk-llmbda",
# ]
#
# [tool.uv.sources]
# tk-llmbda = { path = "../", editable = true }
# ///
from __future__ import annotations

import re

from openai import OpenAI

from tk.llmbda import Skill, SkillContext, StepResult, lm, run_skill

client = OpenAI()

_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def extract_date_regex(ctx: SkillContext) -> StepResult:
    """Pull an ISO-8601 date via regex."""
    if m := _ISO_RE.search(ctx.entry["text"]):
        return StepResult(m.group(1), {"source": "regex"}, resolved=True)
    return StepResult(None, {"reason": "no_iso_date"})


def oai(*, messages, **kwargs):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        **kwargs,
    )
    return resp.choices[0].message.content


@lm(oai, system_prompt="Extract a date from the text. Return ONLY an ISO-8601 date.")
def extract_date_lm(ctx: SkillContext, call) -> StepResult:
    """Extract a date via LLM."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(raw.strip(), {"source": "lm"})


@lm(
    oai,
    system_prompt=(
        "The previous extraction attempt was not a valid ISO-8601 date. "
        "Re-read the original text and try again. Return ONLY YYYY-MM-DD."
    ),
)
def refine_date(ctx: SkillContext, call) -> StepResult:
    """Validate the extracted date; retry via LLM up to 3 times if invalid."""
    prev = ctx.trace.get("ψ::extract") or ctx.trace.get("λ::date")
    value = prev.value if prev else None
    for _ in range(3):
        if value and _ISO_RE.fullmatch(str(value)):
            return StepResult(value, {"valid": True})
        prompt = f"Text: {ctx.entry['text']}\nPrevious attempt: {value}"
        value = call(messages=[{"role": "user", "content": prompt}]).strip()
    return StepResult(value, {"valid": bool(value and _ISO_RE.fullmatch(str(value)))})


skill = Skill(
    name="dates",
    steps=[
        Skill("λ::date", fn=extract_date_regex),
        Skill("ψ::extract", fn=extract_date_lm),
        Skill("ψ::refine", fn=refine_date),
    ],
)

result = run_skill(skill, {"text": "let's meet on the 15th of January 2025"})
print(f"resolved_by: {result.resolved_by}")
print(f"value:       {result.value}")
# resolved_by: ('ψ::refine',)
# value:       2025-01-15
