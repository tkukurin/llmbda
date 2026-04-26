# %% [markdown]
# # Sketch: decorator-as-HOF with `LMCaller` protocol
#
# The `@lm` decorator is a genuine higher-order function: it wraps
# the step function so that `ctx.caller` already has the system prompt
# prepended. The runtime doesn't branch — it just calls `step(ctx)`.
#
# | Authoring | `ctx.caller` the function sees |
# |---|---|
# | bare function | raw caller (or noop — step shouldn't use it) |
# | `@lm()` | raw caller passed through |
# | `@lm(system_prompt=...)` | caller with system prompt prepended |

# %%
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol, runtime_checkable

# %% [markdown]
# ## LMCaller protocol
#
# If a step declares a system prompt, we *know* the caller accepts
# `messages`. The protocol makes that contract explicit so the
# decorator's bound wrapper is a one-liner.

# %%
@runtime_checkable
class LMCaller(Protocol):
    def __call__(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...

Caller = Callable[..., Any]

# %% [markdown]
# ## Core types — one Step, one Result

# %%
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
    fn: Callable[[StepContext], StepResult]
    description: str = ""
    def __call__(self, ctx: StepContext) -> StepResult:
        return self.fn(ctx)

# %% [markdown]
# ## The `@lm` decorator
#
# A real higher-order function: wraps `fn` so that when it runs,
# `ctx.caller` is already bound with the system prompt prepended.
# The protocol guarantees `messages` is a keyword argument.

# %%
def lm(system_prompt: str = ""):
    """Mark a step function as LLM-powered, optionally binding a system prompt."""
    def decorator(fn: Callable[[StepContext], StepResult]) -> Callable[[StepContext], StepResult]:
        if not system_prompt:
            return fn
        @wraps(fn)
        def wrapper(ctx: StepContext) -> StepResult:
            raw = ctx.caller
            def bound(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
                return raw(messages=[{"role": "system", "content": system_prompt}, *messages], **kwargs)
            ctx.caller = bound
            return fn(ctx)
        return wrapper
    return decorator

# %% [markdown]
# ## Runtime — no branching, just `step(ctx)`

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
    """The decorator has already prepended VERIFY_PROMPT to ctx.caller."""
    prior_summary = [
        {"name": s.name, "description": s.description, "metadata": ctx.prior[s.name].metadata}
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
# ## LM step without system_prompt — `@lm()` passes through the raw caller

# %%
@lm()
def summarize(ctx: StepContext) -> StepResult:
    """Pass-through: builds its own messages, no system prompt injected."""
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
# ## Assemble and run

# %%
book_meeting = Skill(
    name="book_meeting",
    steps=[
        Step("parse_weekday", parse_weekday, description="Find a weekday name (Mon-Sun)."),
        Step("parse_time", parse_time, description="Find a clock time like 3pm or 15:00."),
        Step("verify", verify, description="Cross-check and fill gaps."),
        Step("summarize", summarize, description="One-sentence summary."),
    ],
)

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
print(f"metadata:    {result.metadata}")
print()

# %% [markdown]
# ## Verify the decorator did its job
#
# The fake caller records what messages it received. The verify step
# should have gotten a system message prepended; summarize should not.

# %%
captured: list[list[dict[str, str]]] = []
call_count = 0

def spy_caller(*, messages: list[dict[str, str]], **_kwargs: Any) -> str:
    global call_count
    captured.append(list(messages))
    call_count += 1
    if call_count == 1:
        return json.dumps({"weekday": "Tuesday", "time": "3:00pm", "notes": "All confirmed."})
    return "Meeting with Alice on Tuesday at 3pm."

run_skill(book_meeting, {"text": "Meet Tuesday at 3pm"}, caller=spy_caller)

print("verify saw:")
for msg in captured[0]:
    print(f"  {msg['role']:8} {msg['content'][:60]}...")

print()
print("summarize saw:")
for msg in captured[1]:
    print(f"  {msg['role']:8} {msg['content'][:60]}...")

# %% [markdown]
# ## Trace

# %%
print()
for name, sr in result.trace.items():
    print(f"  {name:16} resolved={sr.resolved}  meta={sr.metadata}")