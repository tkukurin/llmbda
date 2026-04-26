# %% [markdown]
# # Sketch: decorator-as-HOF with `LMCaller` protocol
#
# One `Step` type. The `@lm` decorator is a genuine higher-order function
# that wraps the step fn to bind `system_prompt` into `ctx.caller` at call
# time. The runtime doesn't branch — it just calls `step(ctx)`.
#
# | Authoring | Caller behaviour |
# |---|---|
# | bare function | Code step — doesn't touch caller |
# | `@lm()` | Pass-through — raw caller forwarded |
# | `@lm(system_prompt=...)` | Bound — system prompt prepended to messages |
#
# `LMCaller` is a protocol that fixes the caller shape: it must accept
# `messages` as a keyword argument. Because `@lm(system_prompt=...)`
# declares that the caller speaks this protocol, the wrapper doesn't
# need a runtime guard — it just prepends and forwards.

# %%
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol, runtime_checkable

# %% [markdown]
# ## Protocols and types

# %%
@runtime_checkable
class LMCaller(Protocol):
    def __call__(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...

Caller = Callable[..., Any]
StepFn = Callable[["StepContext", ], "StepResult"]

@dataclass
class StepResult:
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = True

@dataclass
class StepContext:
    entry: Any
    caller: Caller
    steps: list[Step] = field(default_factory=list)
    prior: dict[str, StepResult] = field(default_factory=dict)

@dataclass
class Step:
    """A named unit of work that produces a StepResult."""
    name: str
    fn: StepFn
    description: str = ""
    def __call__(self, ctx: StepContext) -> StepResult:
        return self.fn(ctx)

# %% [markdown]
# ## The `@lm` decorator
#
# A real higher-order function: it returns a new `StepFn` that intercepts
# `ctx.caller` at call time and prepends the system prompt. If no prompt
# is given, the function passes through unchanged.

# %%
def lm(system_prompt: str = "", *, description: str = "") -> Callable[[StepFn], StepFn]:
    """Mark a step function as needing an LM caller.

    - `system_prompt` non-empty → caller bound with prompt prepended.
    - `system_prompt` empty → raw caller passed through.
    - Attaches `.lm_description` and `.lm_system_prompt` for introspection.
    """
    def decorator(fn: StepFn) -> StepFn:
        if system_prompt:
            @wraps(fn)
            def wrapper(ctx: StepContext) -> StepResult:
                raw = ctx.caller
                def bound(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
                    return raw(messages=[{"role": "system", "content": system_prompt}, *messages], **kwargs)
                ctx.caller = bound
                return fn(ctx)
            out = wrapper
        else:
            out = fn
        out.lm_system_prompt = system_prompt  # type: ignore[attr-defined]
        out.lm_description = description or system_prompt  # type: ignore[attr-defined]
        return out
    return decorator

# %% [markdown]
# ## Runtime
#
# No branching on step kind. The decorator already handled caller binding
# inside the wrapped function. The runtime just calls `step(ctx)`.

# %%
@dataclass
class Skill:
    name: str
    steps: list[Step] = field(default_factory=list)

@dataclass
class SkillResult:
    skill: str
    resolved_by: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, StepResult] = field(default_factory=dict)

def run_skill(skill: Skill, entry: Any, caller: Caller) -> SkillResult:
    ctx = StepContext(entry=entry, caller=caller, steps=skill.steps)
    trace: dict[str, StepResult] = {}
    last_name, last_result = "(empty)", None
    for i, s in enumerate(skill.steps):
        ctx.caller = caller
        result = s(ctx)
        trace[s.name] = result
        ctx.prior[s.name] = result
        last_name, last_result = s.name, result
        if result.resolved or i == len(skill.steps) - 1:
            break
    if last_result is None:
        return SkillResult(skill=skill.name, resolved_by="(empty)", value=None)
    return SkillResult(
        skill=skill.name,
        resolved_by=last_name,
        value=last_result.value,
        metadata=last_result.metadata,
        trace=trace,
    )

# %% [markdown]
# ## Code steps — plain functions, no decorator

# %%
def parse_weekday(ctx: StepContext) -> StepResult:
    text = ctx.entry["text"].lower()
    days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    for d in days:
        if re.search(rf"\b{d}\b", text):
            return StepResult(value=None, metadata={"weekday": d.capitalize()}, resolved=False)
    return StepResult(value=None, metadata={"weekday": None}, resolved=False)

def parse_time(ctx: StepContext) -> StepResult:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", ctx.entry["text"], re.IGNORECASE)
    if not m:
        return StepResult(value=None, metadata={"time": None}, resolved=False)
    h, mins, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    return StepResult(value=None, metadata={"time": f"{h}:{mins:02d}{ampm}"}, resolved=False)

# %% [markdown]
# ## LM step with system_prompt — `@lm(system_prompt=...)` binds the caller

# %%
VERIFY_PROMPT = """\
You are a calendar booking verifier.
Cross-check the prior parser outputs against the raw text.
Return JSON: {"weekday": ..., "time": ..., "notes": "..."}"""

@lm(system_prompt=VERIFY_PROMPT)
def verify(ctx: StepContext) -> StepResult:
    """The decorator has already bound ctx.caller with VERIFY_PROMPT."""
    prior_summary = [
        {
            "name": s.name,
            "description": s.description or getattr(s.fn, "lm_description", ""),
            "metadata": ctx.prior[s.name].metadata,
        }
        for s in ctx.steps if s.name in ctx.prior
    ]
    payload = json.dumps({"text": ctx.entry["text"], "prior": prior_summary}, indent=2)
    raw = ctx.caller(messages=[{"role": "user", "content": payload}])
    parsed = json.loads(raw)
    return StepResult(
        value={"weekday": parsed.get("weekday"), "time": parsed.get("time")},
        metadata={"notes": parsed.get("notes", "")},
        resolved=False,
    )

# %% [markdown]
# ## LM step without system_prompt — `@lm()` passes through raw caller

# %%
@lm(description="Summarize the booking in one sentence.")
def summarize(ctx: StepContext) -> StepResult:
    """Pass-through: builds its own messages, caller is untouched."""
    booking = ctx.prior.get("verify")
    if not booking or not booking.value:
        return StepResult(value="Nothing to summarize.", metadata={"skipped": True})
    raw = ctx.caller(
        messages=[
            {"role": "system", "content": "Summarize this booking in one sentence."},
            {"role": "user", "content": json.dumps(booking.value)},
        ],
    )
    return StepResult(value=raw, metadata={"source": "pass-through caller"})

# %% [markdown]
# ## Assemble the skill

# %%
book_meeting = Skill(
    name="book_meeting",
    steps=[
        Step("parse_weekday", parse_weekday, description="Find a weekday name (Mon-Sun)."),
        Step("parse_time", parse_time, description="Find a clock time like 3pm or 15:00."),
        Step("verify", verify),
        Step("summarize", summarize),
    ],
)

# %% [markdown]
# ## Run with a fake caller

# %%
call_count = 0

def fake_caller(*, messages: list[dict[str, str]], **_kwargs: Any) -> str:
    global call_count
    call_count += 1
    if call_count == 1:
        return json.dumps({"weekday": "Tuesday", "time": "3:00pm", "notes": "All confirmed."})
    return "Meeting with Alice on Tuesday at 3pm."

result = run_skill(book_meeting, {"text": "Meet Tuesday at 3pm"}, caller=fake_caller)

print(f"resolved_by: {result.resolved_by}")
print(f"value:       {result.value}")
print(f"notes:       {result.metadata}")
print()

# %% [markdown]
# ## Inspect: one Step type, decorator metadata visible via introspection

# %%
for s in book_meeting.steps:
    prompt = getattr(s.fn, "lm_system_prompt", None)
    desc = getattr(s.fn, "lm_description", None)
    if prompt is None:
        binding = "none (pure code)"
    elif prompt:
        binding = "bound to system_prompt"
    else:
        binding = "raw pass-through"
    print(f"  {s.name:16}  caller={binding}")
    if desc:
        print(f"  {'':16}  lm_description={desc[:60]!r}...")

print()
print("trace:")
for name, sr in result.trace.items():
    print(f"  {name:16} resolved={sr.resolved}  meta={sr.metadata}")

# %% [markdown]
# ## Verify: the decorator is a real HOF
#
# The system prompt appears in the caller's messages without the step
# function or the runtime doing anything. The wrapper did it.

# %%
captured: list[list[dict[str, str]]] = []

def spy_caller(*, messages: list[dict[str, str]], **_kwargs: Any) -> str:
    captured.append(messages)
    return json.dumps({"weekday": "Friday", "time": "9:00am", "notes": "ok"})

spy_skill = Skill(
    name="spy",
    steps=[
        Step("parse_weekday", parse_weekday),
        Step("verify", verify),
    ],
)
run_skill(spy_skill, {"text": "Meet Friday at 9am"}, caller=spy_caller)

print("\nMessages the caller actually received:")
for i, msgs in enumerate(captured):
    print(f"  call {i}:")
    for m in msgs:
        role, content = m["role"], m["content"]
        preview = content[:80].replace("\n", "\\n")
        print(f"    {role}: {preview}...")
print()
print("The system prompt was prepended by the @lm decorator, not by the runtime.")