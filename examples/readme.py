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

from tk.llmbda import Skill, Step, StepContext, StepResult, lm, run_skill

client = OpenAI()

_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def extract_date_regex(ctx: StepContext) -> StepResult:
    """pull an ISO-8601 date via regex."""
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


@lm(oai, system_prompt="Extract a date. Return ISO format.")
def extract_date_lm(ctx: StepContext, call) -> StepResult:
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(raw, {"source": "lm"}, resolved=False)


@lm(
    oai,
    system_prompt=(
        "You will receive the original text and prior extraction "
        "attempts. Confirm or correct the date. Return ONLY an ISO-8601 date string."
    ),
)
def verify_date(ctx: StepContext, call) -> StepResult:
    """Cross-check prior date extractions against the raw text."""
    prior_summary = "\n".join(
        f"- {s.name}: value={ctx.prior[s.name].value}, "
        f"meta={ctx.prior[s.name].metadata}"
        for s in ctx.steps
        if s.name in ctx.prior
    )
    prompt = f"Text: {ctx.entry['text']}\n\nPrior steps:\n{prior_summary}"
    raw = call(messages=[{"role": "user", "content": prompt}])
    return StepResult(raw.strip())


skill = Skill(
    name="dates",
    steps=[
        Step("λ::date", extract_date_regex),
        Step("ψ::date", extract_date_lm),
        Step("ψ::date.verify", verify_date),
    ],
)

result = run_skill(skill, {"text": "let's meet on the 15th of January 2025"})
print(f"resolved_by: {result.resolved_by}")
print(f"value:       {result.value}")
# resolved_by: verify_date
# value:       2025-01-15
