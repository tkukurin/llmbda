# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm>=1.0",
#     "tk-llmbda",
# ]
#
# [tool.uv.sources]
# tk-llmbda = { path = "../", editable = true }
# ///
# %% [markdown]
# # Compiled `skill.md` vs programmatic `run_skill`
#
# Side-by-side comparison of two execution paths for the same skill:
#
# - **Programmatic** — `run_skill` executes deterministic fns as Python
#   and routes LLM steps through litellm.
# - **Compiled** — `compile_skill` produces a markdown document.
#   Deterministic leaves become tool-call functions the LLM can invoke;
#   LLM-step prompts are baked into the document the model reads.
#   The model orchestrates everything from the markdown alone.

# %%
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from litellm import completion

from tk.llmbda import (
    LMCaller,
    Skill,
    SkillContext,
    StepResult,
    lm,
    run_skill,
    strip_fences,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compile_sketch import _collect_leaves, _is_lm, _sanitize, compile_skill

# %% [markdown]
# ## Shared setup

# %%
MODEL = "gpt-4o-mini"


def call_lm(*, messages: list[dict[str, str]], **kwargs: Any) -> str:
    resp = completion(model=MODEL, messages=messages, **kwargs)
    return resp.choices[0].message.content or ""


# %% [markdown]
# ## Define the skill (calendar booking)

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

_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?(?:\s*-\s*(\d{1,2})(?::(\d{2}))?)?\s*(am|pm)?\b",
    re.IGNORECASE,
)
_DUR_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minute|min|mins)s?\b",
    re.IGNORECASE,
)
_TOPIC_RE = re.compile(r"(?:about|re:)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE)


def parse_weekday(ctx: SkillContext) -> StepResult:
    """Find an explicit weekday name (Monday..Sunday)."""
    text = ctx.entry["text"].lower()
    for day in WEEKDAYS:
        if re.search(rf"\b{day}\b", text):
            return StepResult(day.capitalize(), {"reason": "matched"})
    return StepResult(None, {"reason": "no_weekday"})


def _fmt(h: int, m: int, ampm: str | None) -> str:
    return f"{h}:{m:02d}{ampm.lower()}" if ampm else f"{h:02d}:{m:02d}"


def parse_time(ctx: SkillContext) -> StepResult:
    """Find a clock time like '3pm', '15:00', or a range '9-10am'."""
    m = _TIME_RE.search(ctx.entry["text"])
    if not m:
        return StepResult(None, {"reason": "no_time"})
    h1, min1, h2, min2, ampm = m.groups()
    start = _fmt(int(h1), int(min1 or 0), ampm)
    end = _fmt(int(h2), int(min2 or 0), ampm) if h2 else None
    return StepResult({"start": start, "end": end}, {"range": bool(end)})


def parse_duration(ctx: SkillContext) -> StepResult:
    """Find a duration phrase like '30 minutes' or '2 hrs' and return minutes."""
    m = _DUR_RE.search(ctx.entry["text"])
    if not m:
        return StepResult(None, {"reason": "no_duration"})
    n, unit = float(m.group(1)), m.group(2).lower()
    minutes = int(n * 60) if unit.startswith(("hour", "hr")) else int(n)
    return StepResult(minutes, {"reason": "matched"})


def parse_topic(ctx: SkillContext) -> StepResult:
    """Find a topic phrase introduced by 'about' or 're:'."""
    m = _TOPIC_RE.search(ctx.entry["text"])
    if not m:
        return StepResult(None, {"reason": "no_topic"})
    return StepResult(m.group(1).strip(), {"reason": "matched"})


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


def _prior_payload(
    trace: dict[str, StepResult],
    skills: list[Skill],
) -> list[dict[str, object]]:
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


@lm(call_lm, system_prompt=VERIFY_PROMPT)
def verify(ctx: SkillContext, steps: list[Skill], call: LMCaller) -> StepResult:
    """Cross-check parser extractions against the raw text."""
    inner = Skill(name="_parse", steps=steps)
    r = run_skill(inner, ctx.entry)
    payload = {
        "text": ctx.entry["text"],
        "prior_steps": _prior_payload(r.trace, steps),
    }
    raw = call(messages=[{"role": "user", "content": json.dumps(payload, indent=2)}])
    try:
        parsed = json.loads(strip_fences(raw))
        return StepResult(
            parsed.get("booking"),
            {"notes": parsed.get("notes", ""), "llm_raw": raw},
        )
    except Exception as exc:  # noqa: BLE001
        return StepResult(None, {"error": str(exc), "llm_raw": raw})


book_meeting = Skill(
    name="book_meeting",
    fn=verify,
    steps=[
        Skill("λ::weekday", fn=parse_weekday),
        Skill("λ::time", fn=parse_time),
        Skill("λ::duration", fn=parse_duration),
        Skill("λ::topic", fn=parse_topic),
    ],
)

INPUT_TEXT = "Can we meet on Tuesday at 3pm for 30 mins about the Q4 review?"

# %% [markdown]
# ## Path A — programmatic `run_skill`

# %%
prog = run_skill(book_meeting, text=INPUT_TEXT)

# %% [markdown]
# ## Path B — compiled `skill.md` + tool calling

# %%
md = compile_skill(book_meeting)


def _make_tool_runner(fn, entry):
    def run():
        ctx = SkillContext(entry=entry)
        r = fn(ctx)
        return {"value": r.value, "metadata": r.metadata}

    return run


def build_tools(
    skill: Skill,
    entry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build tool specs and a dispatch table from deterministic leaves."""
    tools: list[dict[str, Any]] = []
    dispatch: dict[str, Any] = {}
    for leaf in _collect_leaves(skill):
        if _is_lm(leaf):
            continue
        name = _sanitize(leaf.name)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": leaf.description or name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
        dispatch[name] = _make_tool_runner(leaf.fn, entry)
    return tools, dispatch


def run_compiled(
    skill_md: str,
    skill: Skill,
    entry: dict[str, Any],
    *,
    model: str = MODEL,
    max_rounds: int = 10,
) -> str:
    """Drive the compiled skill.md via tool calling and return the final answer."""
    tools, dispatch = build_tools(skill, entry)
    preamble = (
        "You are given a skill document that describes a processing pipeline.\n"
        "For each deterministic step, call the corresponding tool.\n"
        "For each LLM step, follow the instructions in the document yourself.\n"
        "After all steps are done, return the final result.\n\n"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": preamble + skill_md},
        {"role": "user", "content": entry["text"]},
    ]
    for _ in range(max_rounds):
        resp = completion(
            model=model,
            messages=messages,
            tools=tools or None,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg.model_dump())
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            if fn_name not in dispatch:
                result = {"error": f"unknown tool {fn_name!r}"}
            else:
                result = dispatch[fn_name]()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )
    return messages[-1].get("content", "(max rounds reached)")


compiled_answer = run_compiled(md, book_meeting, {"text": INPUT_TEXT})

# %% [markdown]
# ## Compare

# %%
SEP = "=" * 60

print(SEP)
print("COMPILED skill.md")
print(SEP)
print(md)

print(SEP)
print(f"INPUT: {INPUT_TEXT}")
print(SEP)

print("\n--- Programmatic (run_skill) ---")
print(json.dumps(prog.value, indent=2))
print(f"  resolved_by: {prog.resolved_by}")
if prog.metadata.get("notes"):
    print(f"  notes: {prog.metadata['notes']}")

print("\n--- Compiled (skill.md + tool calling) ---")
print(compiled_answer)
