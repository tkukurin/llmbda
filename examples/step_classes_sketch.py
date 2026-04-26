# %% [markdown]
# # Sketch: `Step` vs `LMStep` — caller binding semantics
#
# The type distinction encodes what the **runtime does with the caller**:
#
# | Type | `system_prompt` | Runtime behaviour |
# |---|---|---|
# | `Step` | n/a | No caller binding. Pure computation. |
# | `LMStep` | non-empty | Caller bound with system prompt prepended. |
# | `LMStep` | empty | Raw caller passed through (pass-through). |
#
# `description` on the base `Step` is for cross-step context (downstream
# steps reading intent). It is completely decoupled from caller binding.

# %%
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Caller = Callable[..., Any]

# %% [markdown]
# ## Types

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
    """Pure computation. The runtime does not bind a caller."""
    name: str
    fn: Callable[[StepContext], StepResult]
    description: str = ""
    def __call__(self, ctx: StepContext) -> StepResult:
        return self.fn(ctx)

@dataclass
class LMStep(Step):
    """Needs a caller. Runtime binds system_prompt if present, else passes through."""
    system_prompt: str = ""
    def __post_init__(self) -> None:
        if not self.description and self.system_prompt:
            self.description = self.system_prompt

# %% [markdown]
# ## Runtime
#
# Three-way behaviour:
# - `Step` → caller not rebound (step doesn't need it)
# - `LMStep` + `system_prompt` → caller bound with prompt prepended
# - `LMStep` without → raw caller passed through

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

def _bind_caller(raw: Caller, prompt: str) -> Caller:
    def bound(**kwargs: Any) -> Any:
        if "messages" not in kwargs:
            return raw(**kwargs)
        kwargs["messages"] = [
            {"role": "system", "content": prompt},
            *(kwargs["messages"] or []),
        ]
        return raw(**kwargs)
    return bound

_NOOP_CALLER: Caller = lambda **_kw: None

def run_skill(skill: Skill, entry: Any, caller: Caller) -> SkillResult:
    ctx = StepContext(entry=entry, caller=_NOOP_CALLER, steps=skill.steps)
    trace: dict[str, StepResult] = {}
    last_name, last_result = "(empty)", None
    for i, s in enumerate(skill.steps):
        if isinstance(s, LMStep):
            ctx.caller = _bind_caller(caller, s.system_prompt) if s.system_prompt else caller
        else:
            ctx.caller = _NOOP_CALLER
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
# ## Code steps — plain functions, no caller needed

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
# ## LM step with system_prompt — caller bound with prompt

# %%
VERIFY_PROMPT = """\
You are a calendar booking verifier.
Cross-check the prior parser outputs against the raw text.
Return JSON: {"weekday": ..., "time": ..., "notes": "..."}"""

def verify(ctx: StepContext) -> StepResult:
    """LM step: caller is pre-bound with VERIFY_PROMPT by the runtime."""
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
# ## LM step without system_prompt — raw caller passed through

# %%
def summarize(ctx: StepContext) -> StepResult:
    """LM step with pass-through: builds its own messages, no system prompt."""
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
# ## Assemble the skill — all three cases visible

# %%
book_meeting = Skill(
    name="book_meeting",
    steps=[
        Step("parse_weekday", parse_weekday, description="Find a weekday name (Mon-Sun)."),
        Step("parse_time", parse_time, description="Find a clock time like 3pm or 15:00."),
        LMStep("verify", verify, system_prompt=VERIFY_PROMPT),
        LMStep("summarize", summarize),
    ],
)

# %% [markdown]
# ## Run with a fake caller

# %%
call_count = 0

def fake_caller(**kwargs: Any) -> str:
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
# ## Inspect the skill: types, descriptions, and caller behaviour

# %%
for s in book_meeting.steps:
    kind = type(s).__name__
    prompt = getattr(s, "system_prompt", None)
    if kind == "Step":
        binding = "none (pure code)"
    elif prompt:
        binding = "bound to system_prompt"
    else:
        binding = "raw pass-through"
    print(f"  {kind:6}  {s.name:16}  caller={binding}")

print()
print("trace:")
for name, sr in result.trace.items():
    print(f"  {name:16} resolved={sr.resolved}  meta={sr.metadata}")

# %% [markdown]
# ## Verify the runtime contract
#
# Code steps get a noop caller — calling it is a no-op (returns None).
# This makes accidental caller usage in code steps harmless but visible.

# %%
def code_step_that_tries_caller(ctx: StepContext) -> StepResult:
    got = ctx.caller(messages=[{"role": "user", "content": "hi"}])
    return StepResult(value=f"caller returned: {got!r}")

test_skill = Skill(
    name="test",
    steps=[Step("accidental", code_step_that_tries_caller)],
)
r = run_skill(test_skill, {}, caller=fake_caller)
print(f"\nCode step calling ctx.caller => {r.value}")
print("(noop caller returned None — the real caller was never exposed)")