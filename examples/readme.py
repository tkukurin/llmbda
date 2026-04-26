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

from tk.llmbda import Skill, Step, StepContext, StepResult, lm, loop, run_skill

client = OpenAI()

_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def extract_date_regex(ctx: StepContext) -> StepResult:
    """Pull an ISO-8601 date via regex."""
    if m := _ISO_RE.search(ctx.entry["text"]):
        return StepResult(m.group(1), {"source": "regex"})
    return StepResult(None, {"reason": "no_iso_date"}, resolved=False)


def oai(*, messages, **kwargs):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        **kwargs,
    )
    return resp.choices[0].message.content


@lm(oai, system_prompt="Extract a date from the text. Return ONLY an ISO-8601 date.")
def extract_date_lm(ctx: StepContext, call) -> StepResult:
    """Extract a date via LLM."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(raw.strip(), {"source": "lm"}, resolved=False)


def validate_iso(ctx: StepContext) -> StepResult:
    """Check whether the latest extraction is a valid ISO-8601 date."""
    prev = ctx.prior.get("ψ::extract") or ctx.prior.get("λ::date")
    if prev and _ISO_RE.fullmatch(str(prev.value or "")):
        return StepResult(prev.value, {"valid": True})
    return StepResult(None, {"valid": False}, resolved=False)


@lm(
    oai,
    system_prompt=(
        "The previous extraction attempt was not a valid ISO-8601 date. "
        "Re-read the original text and try again. Return ONLY YYYY-MM-DD."
    ),
)
def retry_extract(ctx: StepContext, call) -> StepResult:
    """Re-attempt date extraction after a failed validation."""
    prev = ctx.prior.get("ψ::extract") or ctx.prior.get("λ::date")
    prev_value = prev.value if prev else None
    prompt = f"Text: {ctx.entry['text']}\nPrevious attempt: {prev_value}"
    raw = call(messages=[{"role": "user", "content": prompt}])
    return StepResult(raw.strip(), {"source": "lm_retry"}, resolved=False)


skill = Skill(
    name="dates",
    steps=[
        Step("λ::date", extract_date_regex),
        Step("ψ::extract", extract_date_lm),
        loop(
            Step("λ::validate", validate_iso),
            Step("ψ::retry", retry_extract),
            name="refine",
            max_iter=3,
            until=lambda ctx: bool(
                ctx.prior.get("validate", StepResult(None))
                .metadata.get("valid")
            ),
        ),
    ],
)

result = run_skill(skill, {"text": "let's meet on the 15th of January 2025"})
print(f"resolved_by: {result.resolved_by}")
print(f"value:       {result.value}")
# resolved_by: validate
# value:       2025-01-15
