"""tk.llmbda -- skill composition for LLMs.

Like lambda calculus, but the functions talk back.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, NamedTuple

Caller = Callable[..., Any]

@dataclass
class StepResult:
    """What a single step produces. ``resolved=False`` falls to the next step."""
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = True

@dataclass
class StepContext:
    """Accumulator threaded through the step sequence."""
    entry: dict[str, Any]
    caller: Caller
    prior: dict[str, StepResult] = field(default_factory=dict)

class Step(NamedTuple):
    """A named step: a label for ``ctx.prior`` plus the callable that runs."""
    name: str
    fn: Callable[[StepContext], StepResult]

@dataclass
class Skill:
    """A named sequence of steps with a shared system prompt."""
    name: str
    steps: list[Step] = field(default_factory=list)
    system_prompt: str = ""

@dataclass
class SkillResult:
    """What a skill execution produces, with a trace of every step that ran."""
    skill: str
    resolved_by: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, StepResult] = field(default_factory=dict)

def iter_skill(
    skill: Skill,
    entry: dict[str, Any],
    caller: Caller,
) -> Iterator[tuple[str, StepResult]]:
    """Yield ``(step_name, result)`` for each executed step, in order."""
    ctx = StepContext(entry=entry, caller=caller)
    last_idx = len(skill.steps) - 1
    for i, step in enumerate(skill.steps):
        result = step.fn(ctx)
        ctx.prior[step.name] = result
        yield step.name, result
        if result.resolved or i == last_idx:
            return

def run_skill(
    skill: Skill,
    entry: dict[str, Any],
    caller: Caller,
) -> SkillResult:
    """Execute *skill* to completion and return the final ``SkillResult``."""
    trace: dict[str, StepResult] = {}
    last: StepResult | None = None
    last_name = "(empty)"
    for name, result in iter_skill(skill, entry, caller):
        trace[name] = result
        last, last_name = result, name

    if last is None:
        return SkillResult(skill=skill.name, resolved_by="(empty)", value=None)
    return SkillResult(
        skill=skill.name,
        resolved_by=last_name,
        value=last.value,
        metadata=last.metadata,
        trace=trace,
    )

_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*\n?(.*?)\n?\s*```$",
    re.DOTALL,
)

def strip_fences(text: str) -> str:
    """Remove Markdown code fences from *text*, if any."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text

__all__ = [
    "Caller",
    "Skill",
    "SkillResult",
    "Step",
    "StepContext",
    "StepResult",
    "iter_skill",
    "run_skill",
    "strip_fences",
]
