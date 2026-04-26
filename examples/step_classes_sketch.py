# %% [markdown]
# # Sketch: `@lm(model)` — decorator binds model at definition time
#
# The `@lm` decorator captures both the model and an optional system prompt.
# The runtime doesn't branch or bind — it just calls `step(ctx)`.
#
# | Authoring | How the function gets its model |
# |---|---|
# | bare function | It doesn't — pure computation |
# | `@lm(model)` | Receives bound caller as second arg |
# | `@lm(model, system_prompt=...)` | Same, with system prompt prepended |
#
# Key consequence: the model must exist before the function is decorated.
# The function carries its own model dependency. `run_skill` doesn't
# take a caller at all — the decorated functions already know their model.

# %%
from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol

# %% [markdown]
# ## Types and decorator

# %%
class LMCaller(Protocol):
    def __call__(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...

StepFn = Callable[["StepContext"], "StepResult"]

@dataclass
class StepResult:
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = True

@dataclass
class StepContext:
    entry: Any
    steps: list[Step] = field(default_factory=list)
    prior: dict[str, StepResult] = field(default_factory=dict)

@dataclass
class Step:
    """A named unit of work that produces a StepResult."""
    name: str
    fn: StepFn
    description: str = ""
    def __post_init__(self) -> None:
        if not self.description:
            self.description = inspect.getdoc(self.fn) or ""
    def __call__(self, ctx: StepContext) -> StepResult:
        return self.fn(ctx)

def lm(model: LMCaller, *, system_prompt: str = "") -> Callable:
    """Bind a step function to a model, optionally prepending a system prompt.

    The inner function receives `(ctx, call)` where `call` is the bound
    caller. The wrapper adapts it to the standard `(ctx) -> StepResult`
    signature that `Step` expects.
    """
    def decorator(fn: Callable) -> StepFn:
        @wraps(fn)
        def wrapper(ctx: StepContext) -> StepResult:
            def bound(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
                if system_prompt:
                    messages = [{"role": "system", "content": system_prompt}, *messages]
                return model(messages=messages, **kwargs)
            return fn(ctx, bound)
        wrapper.lm_system_prompt = system_prompt  # type: ignore[attr-defined]
        wrapper.lm_model = model  # type: ignore[attr-defined]
        return wrapper
    return decorator

# %% [markdown]
# ## Runtime
#
# No branching, no caller parameter. The decorator already handled model
# binding inside the wrapped function. The runtime just calls `step(ctx)`.

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

def run_skill(skill: Skill, entry: Any) -> SkillResult:
    ctx = StepContext(entry=entry, steps=skill.steps)
    trace: dict[str, StepResult] = {}
    last_name, last_result = "(empty)", None
    for i, s in enumerate(skill.steps):
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
#
# Docstrings become `Step.description` automatically via `__post_init__`.

# %%
def parse_weekday(ctx: StepContext) -> StepResult:
    """Find a weekday name (Mon-Sun) in the text."""
    text = ctx.entry["text"].lower()
    days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    for d in days:
        if re.search(rf"\b{d}\b", text):
            return StepResult(value=None, metadata={"weekday": d.capitalize()}, resolved=False)
    return StepResult(value=None, metadata={"weekday": None}, resolved=False)

def parse_time(ctx: StepContext) -> StepResult:
    """Find a clock time like 3pm or 15:00."""
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", ctx.entry["text"], re.IGNORECASE)
    if not m:
        return StepResult(value=None, metadata={"time": None}, resolved=False)
    h, mins, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    return StepResult(value=None, metadata={"time": f"{h}:{mins:02d}{ampm}"}, resolved=False)

# %% [markdown]
# ## Model must exist before decoration
#
# This inverts the usual "define functions, inject caller at runtime" flow.
# The fake caller is defined first so `@lm(fake_caller, ...)` can capture it.

# %%
call_count = 0

def fake_caller(*, messages: list[dict[str, str]], **_kwargs: Any) -> str:
    global call_count
    call_count += 1
    if call_count == 1:
        return json.dumps({"weekday": "Tuesday", "time": "3:00pm", "notes": "All confirmed."})
    return "Meeting with Alice on Tuesday at 3pm."

# %% [markdown]
# ## LM steps — `@lm(model)` binds the model at decoration time
#
# The decorated function receives `(ctx, call)`. The wrapper adapts it
# to the `(ctx) -> StepResult` signature that `Step` expects. No `caller`
# on `StepContext`, no mutation, no restore.

# %%
VERIFY_PROMPT = """\
You are a calendar booking verifier.
Cross-check the prior parser outputs against the raw text.
Return JSON: {"weekday": ..., "time": ..., "notes": "..."}"""

@lm(fake_caller, system_prompt=VERIFY_PROMPT)
def verify(ctx: StepContext, call: LMCaller) -> StepResult:
    """Cross-check parser outputs against the raw text."""
    prior_summary = [  # system_prompt reachable via getattr(s.fn, "lm_system_prompt", "") if needed
        {"name": s.name, "description": s.description, "metadata": ctx.prior[s.name].metadata}
        for s in ctx.steps if s.name in ctx.prior
    ]
    payload = json.dumps({"text": ctx.entry["text"], "prior": prior_summary}, indent=2)
    raw = call(messages=[{"role": "user", "content": payload}])
    parsed = json.loads(raw)
    return StepResult(
        value={"weekday": parsed.get("weekday"), "time": parsed.get("time")},
        metadata={"notes": parsed.get("notes", "")},
        resolved=False,
    )

@lm(fake_caller)
def summarize(ctx: StepContext, call: LMCaller) -> StepResult:
    """Summarize the booking in one sentence."""
    booking = ctx.prior.get("verify")
    if not booking or not booking.value:
        return StepResult(value="Nothing to summarize.", metadata={"skipped": True})
    raw = call(
        messages=[
            {"role": "system", "content": "Summarize this booking in one sentence."},
            {"role": "user", "content": json.dumps(booking.value)},
        ],
    )
    return StepResult(value=raw, metadata={"source": "pass-through caller"})

# %% [markdown]
# ## Assemble the skill
#
# No explicit `description=` needed — `Step.__post_init__` pulls from
# the docstring. `@wraps` copies `__doc__` through the decorator.

# %%
book_meeting = Skill(
    name="book_meeting",
    steps=[
        Step("parse_weekday", parse_weekday),
        Step("parse_time", parse_time),
        Step("verify", verify),
        Step("summarize", summarize),
    ],
)

# %% [markdown]
# ## Run

# %%
result = run_skill(book_meeting, {"text": "Meet Tuesday at 3pm"})

print(f"resolved_by: {result.resolved_by}")
print(f"value:       {result.value}")
print(f"notes:       {result.metadata}")
print()

# %% [markdown]
# ## Introspection
#
# - `s.description` comes from docstrings via `__post_init__`
# - `s.fn.lm_system_prompt` / `s.fn.lm_model` come from the decorator

# %%
for s in book_meeting.steps:
    prompt = getattr(s.fn, "lm_system_prompt", None)
    model = getattr(s.fn, "lm_model", None)
    print(f"  {s.name:16}  desc={s.description!r}")
    if prompt is not None:
        print(f"  {'':16}  system_prompt={'(set)' if prompt else '(empty)'}")
    if model is not None:
        print(f"  {'':16}  model={model.__name__}")

print()
print("trace:")
for name, sr in result.trace.items():
    print(f"  {name:16} resolved={sr.resolved}  meta={sr.metadata}")

# %% [markdown]
# ## Re-binding for testing
#
# `@wraps` sets `__wrapped__`, so you can re-decorate the original
# function body with a different model. Same function, different binding.

# %%
captured: list[list[dict[str, str]]] = []

def spy_caller(*, messages: list[dict[str, str]], **_kwargs: Any) -> str:
    captured.append(messages)
    return json.dumps({"weekday": "Friday", "time": "9:00am", "notes": "ok"})

verify_spy = lm(spy_caller, system_prompt=VERIFY_PROMPT)(verify.__wrapped__)

spy_skill = Skill(
    name="spy",
    steps=[
        Step("parse_weekday", parse_weekday),
        Step("verify", verify_spy),
    ],
)
run_skill(spy_skill, {"text": "Meet Friday at 9am"})

print("\nMessages the spy caller received:")
for i, msgs in enumerate(captured):
    print(f"  call {i}:")
    for m in msgs:
        role, content = m["role"], m["content"]
        preview = content[:80].replace("\n", "\\n")
        print(f"    {role}: {preview}...")
print()
print("System prompt prepended by @lm, model bound at decoration time.")