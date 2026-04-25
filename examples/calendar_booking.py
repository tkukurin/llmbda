# %% [markdown]
# # Extracting a calendar booking from a natural-language request
#
# A realistic `tk.llmbda` pipeline where several deterministic parsers each
# extract a partial piece of information, then a final LLM step **verifies
# and fills in** the picture by looking at the raw text plus what every
# earlier step found (and what each earlier step was *trying* to find).
#
# The pattern:
#
# 1. `parse_weekday` — regex for weekday names.
# 2. `parse_time` — regex for explicit clock times (incl. ranges like `9-10am`).
# 3. `parse_duration` — regex for duration phrases (`30 mins`, `2 hours`).
# 4. `parse_topic` — regex for `"about|re: <topic>"` snippets.
# 5. `llm_verify_and_fill` — reads `ctx.prior` (extractions) and `ctx.steps`
#    (each step's `system_prompt`, i.e. its intent) and produces a final
#    `Booking` dict or a structured diagnosis.
#
# Intermediate steps return `resolved=False` so the runtime keeps walking the
# chain; they stash what they found in `metadata` rather than `value`. The
# verifier is the terminal step and is what `SkillResult.resolved_by` will
# point to.
#
# The notebook runs end-to-end without an API key — a scripted fake caller
# returns plausible LLM responses. A final cell shows how to swap in a real
# OpenAI client.

# %%
from __future__ import annotations

import json
import re

from tk.llmbda import Skill, Step, StepContext, StepResult, run_skill, strip_fences

# %% [markdown]
# ## Step 1 — weekday
#
# A thin regex over the seven weekday names. We stash the match (or its
# absence) in `metadata` and always fall through. The `system_prompt`
# captures the step's intent so the verifier can read it back.

# %%
WEEKDAY_PROMPT = (
    "Find an explicit weekday name (Monday..Sunday, case-insensitive) "
    "and return it capitalised."
)

WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
)

def parse_weekday(ctx: StepContext) -> StepResult:
    """Regex-find a weekday in `ctx.entry['text']`, stash in metadata."""
    text = ctx.entry["text"].lower()
    for day in WEEKDAYS:
        if re.search(rf"\b{day}\b", text):
            return StepResult(
                value=None,
                metadata={"weekday": day.capitalize(), "reason": "matched"},
                resolved=False,
            )
    return StepResult(
        value=None,
        metadata={"weekday": None, "reason": "no_weekday_in_text"},
        resolved=False,
    )

# %%
parse_weekday(
    StepContext(entry={"text": "meet Tuesday at 3pm"}, caller=lambda **_: None),
).metadata

# %% [markdown]
# ## Step 2 — time
#
# Handles `"3pm"`, `"15:00"`, and ranges like `"9-10am"` (in which case we
# stash both start and end so duration can be inferred downstream).

# %%
TIME_PROMPT = (
    "Find a clock time like '3pm', '15:00', or a range '9-10am'. "
    "When a range is given, return both start and end."
)

_TIME_RE = re.compile(
    r"""
    \b(\d{1,2})                 # hour
    (?::(\d{2}))?               # :minutes
    (?:\s*-\s*(\d{1,2})(?::(\d{2}))?)?  # optional -end
    \s*(am|pm)?\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _fmt(h: int, m: int, ampm: str | None) -> str:
    if ampm:
        return f"{h}:{m:02d}{ampm.lower()}"
    return f"{h:02d}:{m:02d}"

def parse_time(ctx: StepContext) -> StepResult:
    """Regex-find a clock time or time range; stash start (and end) in metadata."""
    text = ctx.entry["text"]
    m = _TIME_RE.search(text)
    if not m:
        return StepResult(
            value=None,
            metadata={"start": None, "end": None, "reason": "no_time_found"},
            resolved=False,
        )
    h1, min1, h2, min2, ampm = m.groups()
    start = _fmt(int(h1), int(min1 or 0), ampm)
    end = _fmt(int(h2), int(min2 or 0), ampm) if h2 else None
    return StepResult(
        value=None,
        metadata={
            "start": start,
            "end": end,
            "reason": "matched_range" if end else "matched_single",
        },
        resolved=False,
    )

# %%
for text in ("3pm", "meeting 9-10am", "at 15:00", "no time here"):
    r = parse_time(StepContext(entry={"text": text}, caller=lambda **_: None))
    print(f"{text!r:30} -> {r.metadata}")

# %% [markdown]
# ## Step 3 — duration

# %%
DURATION_PROMPT = (
    "Find a duration phrase like '30 minutes', '2 hrs', 'half an hour' "
    "and return it in minutes. Only handle explicit numeric forms here."
)

_DUR_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minute|min|mins)s?\b",
    re.IGNORECASE,
)

def parse_duration(ctx: StepContext) -> StepResult:
    text = ctx.entry["text"]
    m = _DUR_RE.search(text)
    if not m:
        return StepResult(
            value=None,
            metadata={"minutes": None, "reason": "no_duration"},
            resolved=False,
        )
    n, unit = float(m.group(1)), m.group(2).lower()
    minutes = int(n * 60) if unit.startswith(("hour", "hr")) else int(n)
    return StepResult(
        value=None,
        metadata={"minutes": minutes, "reason": "matched"},
        resolved=False,
    )

# %% [markdown]
# ## Step 4 — topic
#
# Grabs whatever follows "about" or "re:" up to end-of-sentence.

# %%
TOPIC_PROMPT = (
    "Find a topic phrase introduced by 'about' or 're:' and return the "
    "topic as a short noun phrase."
)

_TOPIC_RE = re.compile(r"(?:about|re:)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE)

def parse_topic(ctx: StepContext) -> StepResult:
    m = _TOPIC_RE.search(ctx.entry["text"])
    if not m:
        return StepResult(
            value=None,
            metadata={"topic": None, "reason": "no_topic_marker"},
            resolved=False,
        )
    return StepResult(
        value=None,
        metadata={"topic": m.group(1).strip(), "reason": "matched"},
        resolved=False,
    )

# %% [markdown]
# ## Step 5 — LLM verifier & filler
#
# This is where `ctx.prior` + `ctx.steps` earn their keep. The step builds a
# user-message that shows the model:
#
# - the original `text`,
# - for every prior step: its `name`, its `system_prompt` (what it was looking
#   for), and its `metadata` (what it found, or why it didn't).
#
# The model is asked to return a final `Booking`, *cross-checking* each prior
# finding against the raw text. If a regex step said "no weekday" but the text
# says "tomorrow", the verifier owns that resolution — and it can also disagree
# with a regex step that over-matched.

# %%
VERIFY_PROMPT = """\
You are a calendar booking verifier.
You receive a JSON object with:
- "text": the original request.
- "prior_steps": each earlier parser with {name, system_prompt, metadata}.

Your job is to produce a final booking by cross-checking the prior_steps
against the raw text. You MUST:
- confirm each prior finding against the text, or correct it,
- fill any gaps using the text alone (do not invent),
- flag anything the user left ambiguous.

Return ONLY JSON:
{
  "booking": {
    "weekday":  "<Monday..Sunday or null>",
    "start":    "<HH:MM or H[:MM]am/pm or null>",
    "end":      "<same format, or null>",
    "minutes":  <int or null>,
    "topic":    "<short string or null>"
  },
  "notes": "<one sentence: what you confirmed, corrected, or flagged>"
}
"""

def _prior_steps_payload(ctx: StepContext) -> list[dict[str, object]]:
    """Name + intent + outcome for each step that has already run."""
    return [
        {
            "name": s.name,
            "system_prompt": s.system_prompt,
            "metadata": ctx.prior[s.name].metadata,
        }
        for s in ctx.steps
        if s.name in ctx.prior
    ]

def llm_verify_and_fill(ctx: StepContext) -> StepResult:
    """Ask the LLM to cross-check prior extractions and produce a final Booking."""
    payload = {
        "text": ctx.entry["text"],
        "prior_steps": _prior_steps_payload(ctx),
    }
    try:
        raw = ctx.caller(
            messages=[
                {"role": "system", "content": ctx.step.system_prompt},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
        )
        parsed = json.loads(strip_fences(raw))
        return StepResult(
            value=parsed.get("booking"),
            metadata={"notes": parsed.get("notes", ""), "llm_raw": raw},
        )
    except Exception as exc:  # noqa: BLE001 -- verifier must survive any caller/parse error
        return StepResult(
            value=None,
            metadata={"reason": "llm_parse_error", "error": str(exc)},
        )

# %% [markdown]
# ## Assemble the skill

# %%
book_meeting = Skill(
    name="book_meeting",
    steps=[
        Step("parse_weekday",  parse_weekday,  system_prompt=WEEKDAY_PROMPT),
        Step("parse_time",     parse_time,     system_prompt=TIME_PROMPT),
        Step("parse_duration", parse_duration, system_prompt=DURATION_PROMPT),
        Step("parse_topic",    parse_topic,    system_prompt=TOPIC_PROMPT),
        Step("verify",         llm_verify_and_fill, system_prompt=VERIFY_PROMPT),
    ],
)

# %% [markdown]
# ## A scripted caller so the notebook runs without an API key
#
# The fake caller matches on substrings of the input `text` and returns a
# pre-baked JSON response. This lets you see the verifier payload flow
# through without burning tokens. Swap it for the real caller at the bottom.

# %%
def _canned_response(user_msg: str) -> str:
    """Return a plausible verifier response based on the text snippet."""
    if "Tuesday at 3pm for 30 mins" in user_msg:
        return json.dumps({
            "booking": {
                "weekday": "Tuesday", "start": "3:00pm", "end": None,
                "minutes": 30, "topic": "the Q4 review",
            },
            "notes": "All prior findings confirmed against the text.",
        })
    if "tomorrow at 2" in user_msg:
        return json.dumps({
            "booking": {
                "weekday": None, "start": "2:00pm", "end": None,
                "minutes": 30, "topic": "the launch",
            },
            "notes": (
                "Weekday parser found nothing; 'tomorrow' is relative — "
                "left weekday null. Assumed pm and a default 30-minute slot."
            ),
        })
    if "Friday 9-10am" in user_msg:
        return json.dumps({
            "booking": {
                "weekday": "Friday", "start": "9:00am", "end": "10:00am",
                "minutes": 60, "topic": "hiring",
            },
            "notes": (
                "Duration parser missed the implicit 60 min from the '9-10am' "
                "range; computed it from the time endpoints."
            ),
        })
    if "sync next week" in user_msg:
        return json.dumps({
            "booking": {
                "weekday": None, "start": None, "end": None,
                "minutes": None, "topic": "onboarding",
            },
            "notes": "No day, time, or duration given — user must clarify.",
        })
    return json.dumps({
        "booking": {"weekday": None, "start": None, "end": None,
                    "minutes": None, "topic": None},
        "notes": "No canned response for this input.",
    })

def scripted_caller(**kwargs: object) -> str:
    """Pretend to be an OpenAI caller; returns a JSON string."""
    messages = kwargs["messages"]
    user_msg = messages[1]["content"]  # type: ignore[index]
    return _canned_response(user_msg)

# %% [markdown]
# ## Run it on a handful of realistic requests

# %%
REQUESTS = [
    "Can we meet on Tuesday at 3pm for 30 mins about the Q4 review?",
    "Let's catch up tomorrow at 2 about the launch.",
    "Meeting Friday 9-10am re: hiring.",
    "Let's sync next week about onboarding.",
]

for text in REQUESTS:
    result = run_skill(book_meeting, {"text": text}, caller=scripted_caller)
    print(f"\n--- {text}")
    print(f"resolved_by: {result.resolved_by}")
    print(f"booking:     {result.value}")
    print(f"notes:       {result.metadata.get('notes')}")

# %% [markdown]
# ## Inspect what the verifier actually saw
#
# To understand why this design works, it helps to look at the JSON payload
# the verifier receives. Every prior step shows up with its *intent* plus its
# *outcome* — which is exactly the context needed to cross-check.

# %%
def dump_verifier_payload(text: str) -> None:
    """Run the skill with a spy caller that prints the verifier's user msg."""
    def spy(**kwargs: object) -> str:
        user_msg = kwargs["messages"][1]["content"]  # type: ignore[index]
        print(user_msg)
        return _canned_response(user_msg)
    run_skill(book_meeting, {"text": text}, caller=spy)

dump_verifier_payload("Meeting Friday 9-10am re: hiring.")

# %% [markdown]
# Notice how the verifier's payload says `parse_duration` looked for a
# duration phrase and found none (`minutes: null`), yet the time parser
# found a range (`9-10am`). That's exactly the inconsistency the verifier
# is in a position to resolve — and the `notes` field shows it did.

# %% [markdown]
# ## Walking the trace step-by-step with `iter_skill`
#
# `run_skill` gives you the final result. For observability (or to bail out
# early) use `iter_skill`, which yields each `(step_name, StepResult)` as it
# runs.

# %%
from tk.llmbda import iter_skill

for name, step_result in iter_skill(
    book_meeting,
    {"text": "Meeting Friday 9-10am re: hiring."},
    caller=scripted_caller,
):
    print(f"{name:16} value={step_result.value!r:8} meta={step_result.metadata}")

# %% [markdown]
# ## Swapping in a real OpenAI caller
#
# The caller contract is a keyword-only callable that accepts whatever kwargs
# your steps pass (typically `messages=...`) and returns the model's reply as
# a string. Here's a minimal real caller; uncomment and set `OPENAI_API_KEY`
# to use it.

# %%
# import os, urllib.request
#
# def openai_caller(**kwargs: object) -> str:
#     payload = json.dumps(
#         {"model": "gpt-4o-mini", "temperature": 0.0, **kwargs},
#     ).encode("utf-8")
#     req = urllib.request.Request(
#         "https://api.openai.com/v1/chat/completions",
#         data=payload,
#         headers={
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
#         },
#     )
#     with urllib.request.urlopen(req) as response:
#         body = json.loads(response.read().decode("utf-8"))
#         return body["choices"][0]["message"]["content"]
#
# result = run_skill(
#     book_meeting,
#     {"text": "Meeting Friday 9-10am re: hiring."},
#     caller=openai_caller,
# )
# print(result)
