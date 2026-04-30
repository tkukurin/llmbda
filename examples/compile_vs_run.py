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
# Side-by-side comparison of two execution paths for the same skill.
#
# - **Programmatic** — `run_skill` executes deterministic fns as Python and
#   routes LLM steps through the bound caller.
# - **Compiled** — `compile_skill` produces a markdown document. Deterministic
#   leaves become tool calls; the root prompt is read from the compiled doc.
# - Set `COMPILE_VS_RUN_MODEL` to a tool-capable model to exercise the live
#   compiled path, for example `gpt-4o-mini`.

# %%
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

from litellm import completion

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compile_sketch import _collect_leaves, _is_lm, _sanitize, compile_skill

# %% [markdown]
# ## Shared setup

# %%
TOOL_MODEL = os.environ.get("COMPILE_VS_RUN_MODEL")
INPUT_TEXT = "Can we meet on Tuesday at 3pm for 30 mins about the Q4 review?"


_CANNED_BOOKINGS = {
    "Tuesday at 3pm for 30 mins about the Q4 review": {
        "booking": {
            "weekday": "Tuesday",
            "start": "3:00pm",
            "end": None,
            "minutes": 30,
            "topic": "the Q4 review",
        },
        "notes": "All prior findings confirmed against the text.",
    }
}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "".join(parts)


def scripted_booking_caller(*, messages: list[dict[str, str]], **_kw: object) -> str:
    """Scripted LMCaller for examples; returns a JSON string."""
    user_msg = messages[1]["content"]
    for key, payload in _CANNED_BOOKINGS.items():
        if key in user_msg:
            return json.dumps(payload)
    return json.dumps({"booking": {}, "notes": "No canned response for this input."})


def live_lm(*, messages: list[dict[str, str]], **kwargs: Any) -> str:
    """litellm-backed LMCaller for the programmatic path."""
    assert TOOL_MODEL is not None
    resp = cast("Any", completion(model=TOOL_MODEL, messages=messages, **kwargs))
    return _message_text(resp.choices[0].message.content)


call_lm = live_lm if TOOL_MODEL else scripted_booking_caller

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
            return StepResult(value=day.capitalize(), meta={"reason": "matched"})
    return StepResult(meta={"reason": "no_weekday"})


def _fmt(h: int, m: int, ampm: str | None) -> str:
    return f"{h}:{m:02d}{ampm.lower()}" if ampm else f"{h:02d}:{m:02d}"


def parse_time(ctx: SkillContext) -> StepResult:
    """Find a clock time like '3pm', '15:00', or a range '9-10am'."""
    match = _TIME_RE.search(ctx.entry["text"])
    if not match:
        return StepResult(meta={"reason": "no_time"})
    h1, min1, h2, min2, ampm = match.groups()
    start = _fmt(int(h1), int(min1 or 0), ampm)
    end = _fmt(int(h2), int(min2 or 0), ampm) if h2 else None
    result = {"start": start, "end": end}
    return StepResult(value=result, meta={"range": bool(end)})


def parse_duration(ctx: SkillContext) -> StepResult:
    """Find a duration phrase like '30 minutes' or '2 hrs' and return minutes."""
    match = _DUR_RE.search(ctx.entry["text"])
    if not match:
        return StepResult(meta={"reason": "no_duration"})
    n, unit = float(match.group(1)), match.group(2).lower()
    minutes = int(n * 60) if unit.startswith(("hour", "hr")) else int(n)
    return StepResult(value=minutes, meta={"reason": "matched"})


def parse_topic(ctx: SkillContext) -> StepResult:
    """Find a topic phrase introduced by 'about' or 're:'."""
    match = _TOPIC_RE.search(ctx.entry["text"])
    if not match:
        return StepResult(meta={"reason": "no_topic"})
    return StepResult(value=match.group(1).strip(), meta={"reason": "matched"})


VERIFY_PROMPT = """\
You are a calendar booking verifier.
Input JSON has "text" (original request) and "prior_steps" (each parser's
name/description/value/meta). Cross-check the prior findings against
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
            "name": skill.name,
            "description": skill.description,
            "value": trace[skill.name].value,
            "meta": trace[skill.name].meta,
        }
        for skill in skills
        if skill.name in trace
    ]


@lm(call_lm, system_prompt=VERIFY_PROMPT)
def verify(ctx: SkillContext, steps: list[Skill], call: LMCaller) -> StepResult:
    """Run parser children, cross-check extractions against the raw text."""
    inner = Skill(name="_parse", steps=steps)
    trace = run_skill(inner, ctx.entry)
    payload = {
        "text": ctx.entry["text"],
        "prior_steps": _prior_payload(trace, steps),
    }
    raw = call(messages=[{"role": "user", "content": json.dumps(payload, indent=2)}])
    try:
        parsed = json.loads(strip_fences(raw))
        return StepResult(
            value=parsed.get("booking"),
            meta={"notes": parsed.get("notes", ""), "llm_raw": raw},
        )
    except Exception as exc:  # noqa: BLE001
        return StepResult(meta={"reason": "llm_parse_error", "error": str(exc)})


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

# %% [markdown]
# ## Path A — programmatic `run_skill`

# %%
prog_trace = run_skill(Skill(name="s", steps=[book_meeting]), text=INPUT_TEXT)
prog_result = last(prog_trace)
prog_payload = {
    "booking": prog_result.value,
    "notes": prog_result.meta.get("notes", ""),
}

# %% [markdown]
# ## Path B — compiled `skill.md` + live tool use

# %%
md = compile_skill(book_meeting)


def _make_tool_runner(fn, entry: dict[str, Any]):
    def run() -> dict[str, Any]:
        ctx = SkillContext(entry=entry)
        result = fn(ctx)
        if not isinstance(result, StepResult):
            result = StepResult(value=result)
        return {"value": result.value, "meta": result.meta}

    return run


def build_tools(
    skill: Skill,
    entry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Build tool specs and dispatch from deterministic leaves."""
    tools: list[dict[str, Any]] = []
    dispatch: dict[str, Any] = {}
    lines: list[str] = []
    for leaf in _collect_leaves(skill):
        if _is_lm(leaf):
            continue
        tool_name = _sanitize(leaf.name)
        lines.append(f"- `{tool_name}` executes `{leaf.name}`")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": (
                        f"Execute deterministic step {leaf.name}. "
                        f"{leaf.description or ''}"
                    ).strip(),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        )
        dispatch[tool_name] = _make_tool_runner(leaf.fn, entry)
    return tools, dispatch, lines


def run_compiled(
    skill_md: str,
    skill: Skill,
    entry: dict[str, Any],
    *,
    model: str,
    max_rounds: int = 8,
) -> str:
    """Drive the compiled skill through a live tool-calling model."""
    tools, dispatch, tool_lines = build_tools(skill, entry)
    tool_text = "\n".join(tool_lines)
    preamble = (
        "You are executing a compiled skill document.\n"
        "Call deterministic tools when you need their outputs.\n"
        "The tools already run against the current request state, so do not "
        "invent arguments.\n"
        "Use tool outputs exactly as returned.\n"
        "After using the needed tools, return ONLY the final JSON result.\n\n"
        "Available deterministic tools:\n"
        f"{tool_text}\n\n"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": preamble + skill_md},
        {"role": "user", "content": entry["text"]},
    ]
    for _ in range(max_rounds):
        resp = cast(
            "Any",
            completion(model=model, messages=messages, tools=tools, tool_choice="auto"),
        )
        msg = resp.choices[0].message
        tool_calls = list(msg.tool_calls or [])
        content = _message_text(msg.content)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        messages.append(assistant_msg)
        if not tool_calls:
            return content
        for tc in tool_calls:
            fn_name = tc.function.name
            result = (
                dispatch[fn_name]()
                if fn_name in dispatch
                else {"error": f"unknown tool {fn_name!r}"}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(result),
                }
            )
    msg = f"compiled path exceeded {max_rounds} rounds"
    raise RuntimeError(msg)


compiled_answer = (
    run_compiled(md, book_meeting, {"text": INPUT_TEXT}, model=TOOL_MODEL)
    if TOOL_MODEL
    else None
)

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
print(json.dumps(prog_payload, indent=2))

print("\n--- Compiled (skill.md + tool calling) ---")
if compiled_answer is None:
    print("Set COMPILE_VS_RUN_MODEL to run the live tool-use path.")
else:
    print(compiled_answer)
