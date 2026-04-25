"""tk.llmbda -- skill composition for LLMs.

Like lambda calculus, but the functions talk back.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

Caller = Callable[..., Any]

@dataclass
class StepResult:
    """What a single step produces. ``resolved=False`` falls to the next step."""
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = True

@dataclass
class Step:
    """A named step with an optional system prompt describing its intent.

    The prompt is the step's job description: LLM steps forward it to the model;
    deterministic steps expose it so later LLM steps can include it as context.
    """
    name: str
    fn: Callable[[StepContext], StepResult]
    system_prompt: str = ""

@dataclass
class StepContext:
    """Accumulator threaded through the step sequence.

    ``caller`` is per-step: the runtime rebinds it to prepend the current
    step's ``system_prompt`` to any ``messages=`` kwarg before forwarding.
    """
    entry: Any
    caller: Caller
    steps: list[Step] = field(default_factory=list)
    prior: dict[str, StepResult] = field(default_factory=dict)

@dataclass
class Skill:
    """A named sequence of steps."""
    name: str
    steps: list[Step] = field(default_factory=list)

@dataclass
class SkillResult:
    """What a skill execution produces, with a trace of every step that ran."""
    skill: str
    resolved_by: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, StepResult] = field(default_factory=dict)

def _bind_caller(raw: Caller, step: Step) -> Caller:
    """Wrap *raw* so ``messages=`` kwargs get *step*'s system prompt prepended."""
    if not step.system_prompt:
        return raw
    def bound(**kwargs: Any) -> Any:
        msgs = kwargs.get("messages", [])
        kwargs["messages"] = [
            {"role": "system", "content": step.system_prompt},
            *msgs,
        ]
        return raw(**kwargs)
    return bound

def iter_skill(
    skill: Skill,
    entry: Any,
    caller: Caller,
) -> Iterator[tuple[str, StepResult]]:
    """Yield ``(step_name, result)`` for each executed step, in order.

    The last step is always treated as resolved, so the skill returns something.
    """
    ctx = StepContext(entry=entry, caller=caller, steps=skill.steps)
    last_idx = len(skill.steps) - 1
    for i, step in enumerate(skill.steps):
        ctx.caller = _bind_caller(caller, step)
        result = step.fn(ctx)
        ctx.prior[step.name] = result
        yield step.name, result
        if result.resolved or i == last_idx:
            return

def run_skill(
    skill: Skill,
    entry: Any,
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
