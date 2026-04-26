# %% [markdown]
# # Calendar booking from a natural-language request
#
# Four regex parsers each extract a fragment; an LLM verifier cross-checks
# them against the raw text, fills gaps, and flags ambiguity.
#
# - Intermediate steps return `resolved=False` and fall through.
# - The verifier reads `ctx.prior` (what each step found) and `ctx.steps`
#   (each step's `description`) so it knows both the outcome and the intent.
# - `@lm(model, system_prompt=...)` binds the model at decoration time, so
#   the caller must be defined first.

# %%
from __future__ import annotations

import json
import re

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

# %% [markdown]
# ## Regex parsers

# %%
WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def parse_weekday(ctx: StepContext) -> StepResult:
    """Find an explicit weekday name (Monday..Sunday)."""
    text = ctx.entry["text"].lower()
    for day in WEEKDAYS:
        if re.search(rf"\b{day}\b", text):
            return StepResult(day.capitalize(), {"reason": "matched"}, resolved=False)
    return StepResult(None, {"reason": "no_weekday"}, resolved=False)


# %%
_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?(?:\s*-\s*(\d{1,2})(?::(\d{2}))?)?\s*(am|pm)?\b",
    re.IGNORECASE,
)


def _fmt(h: int, m: int, ampm: str | None) -> str:
    return f"{h}:{m:02d}{ampm.lower()}" if ampm else f"{h:02d}:{m:02d}"


def parse_time(ctx: StepContext) -> StepResult:
    """Find a clock time like '3pm', '15:00', or a range '9-10am'."""
    m = _TIME_RE.search(ctx.entry["text"])
    if not m:
        return StepResult(None, {"reason": "no_time"}, resolved=False)
    h1, min1, h2, min2, ampm = m.groups()
    start = _fmt(int(h1), int(min1 or 0), ampm)
    end = _fmt(int(h2), int(min2 or 0), ampm) if h2 else None
    result = {"start": start, "end": end}
    return StepResult(result, {"range": bool(end)}, resolved=False)


# %%
_DUR_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minute|min|mins)s?\b",
    re.IGNORECASE,
)


def parse_duration(ctx: StepContext) -> StepResult:
    """Find a duration phrase like '30 minutes' or '2 hrs' and return minutes."""
    m = _DUR_RE.search(ctx.entry["text"])
    if not m:
        return StepResult(None, {"reason": "no_duration"}, resolved=False)
    n, unit = float(m.group(1)), m.group(2).lower()
    minutes = int(n * 60) if unit.startswith(("hour", "hr")) else int(n)
    return StepResult(minutes, {"reason": "matched"}, resolved=False)


# %%
_TOPIC_RE = re.compile(r"(?:about|re:)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE)


def parse_topic(ctx: StepContext) -> StepResult:
    """Find a topic phrase introduced by 'about' or 're:'."""
    m = _TOPIC_RE.search(ctx.entry["text"])
    if not m:
        return StepResult(None, {"reason": "no_topic"}, resolved=False)
    return StepResult(m.group(1).strip(), {"reason": "matched"}, resolved=False)


# %% [markdown]
# ## Scripted caller
#
# Defined before the LLM step so `@lm(...)` can capture it at decoration time.
# Real usage swaps this for an OpenAI-style caller (see README).

# %%
_CANNED = {
    "Tuesday at 3pm for 30 mins": {
        "booking": {
            "weekday": "Tuesday",
            "start": "3:00pm",
            "end": None,
            "minutes": 30,
            "topic": "the Q4 review",
        },
        "notes": "All prior findings confirmed against the text.",
    },
    "Friday 9-10am": {
        "booking": {
            "weekday": "Friday",
            "start": "9:00am",
            "end": "10:00am",
            "minutes": 60,
            "topic": "hiring",
        },
        "notes": "Duration parser missed the implicit 60 min from the range.",
    },
}
_DEFAULT = {
    "booking": {
        "weekday": None,
        "start": None,
        "end": None,
        "minutes": None,
        "topic": None,
    },
    "notes": "No canned response for this input.",
}


def scripted_caller(*, messages: list[dict[str, str]], **_kw: object) -> str:
    """Pretend to be an OpenAI caller; returns a JSON string."""
    user_msg = messages[1]["content"]
    for key, payload in _CANNED.items():
        if key in user_msg:
            return json.dumps(payload)
    return json.dumps(_DEFAULT)


# %% [markdown]
# ## LLM verifier
#
# Serialises each prior step's *intent* (`description`) plus *outcome*
# (`value`, `metadata`) and asks the model to produce a final booking.

# %%
VERIFY_PROMPT = """\
You are a calendar booking verifier.
Input JSON has "text" (original request) and "prior_steps" (each parser's
name/description/value/metadata). Cross-check the prior findings against
the text: confirm, correct, fill gaps (no invention), flag ambiguity.

Return ONLY JSON:
{
  "booking": {"weekday": ..., "start": ..., "end": ..., "minutes": ..., "topic": ...},
  "notes": "<one sentence>"
}
"""


def _prior_payload(ctx: StepContext) -> list[dict[str, object]]:
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


@lm(scripted_caller, system_prompt=VERIFY_PROMPT)
def verify(ctx: StepContext, call: LMCaller) -> StepResult:
    """Cross-check prior extractions against the raw text, produce a Booking."""
    payload = {"text": ctx.entry["text"], "prior_steps": _prior_payload(ctx)}
    try:
        msg = json.dumps(payload, indent=2)
        raw = call(messages=[{"role": "user", "content": msg}])
        parsed = json.loads(strip_fences(raw))
        notes = parsed.get("notes", "")
        return StepResult(
            parsed.get("booking"),
            {"notes": notes, "llm_raw": raw},
        )
    except Exception as exc:  # noqa: BLE001
        return StepResult(None, {"reason": "llm_parse_error", "error": str(exc)})


# %% [markdown]
# ## Assemble and run

# %%
book_meeting = Skill(
    name="book_meeting",
    steps=[
        Step("λ::weekday", parse_weekday),
        Step("λ::time", parse_time),
        Step("λ::duration", parse_duration),
        Step("λ::topic", parse_topic),
        Step("ψ::verify", verify),
    ],
)

REQUESTS = [
    "Can we meet on Tuesday at 3pm for 30 mins about the Q4 review?",
    "Meeting Friday 9-10am re: hiring.",
    "Let's sync next week about onboarding.",
]

for text in REQUESTS:
    result = run_skill(book_meeting, {"text": text})
    print(f"\n--- {text}")
    print(f"resolved_by: {result.resolved_by}")
    print(f"booking:     {result.value}")
    print(f"notes:       {result.metadata.get('notes')}")
