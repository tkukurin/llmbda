"""tk.llmbda -- skill composition for LLMs.

Like lambda calculus, but the functions talk back.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class _Msg(Protocol): content: str
class _Choice(Protocol): message: _Msg
class Completion(Protocol): choices: list[_Choice]

Caller = Callable[..., Completion]

@dataclass
class StepResult:
    """What a single step produces."""
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal: bool = False

@dataclass
class StepContext:
    """Accumulator threaded through the step sequence."""
    entry: dict[str, Any]
    caller: Caller | None = None
    prior: dict[str, StepResult] = field(default_factory=dict)

Step = Callable[[StepContext], StepResult]

@dataclass
class Skill:
    """A named sequence of steps with a shared system prompt."""
    name: str
    steps: list[Step] = field(default_factory=list)
    system_prompt: str = ""

def run_skill(
    skill: Skill,
    entry: dict[str, Any],
    caller: Caller | None = None,
) -> StepResult:
    """Execute *skill*'s steps in order and return a ``StepResult``."""
    ctx = StepContext(entry=entry, caller=caller)

    for step in skill.steps:
        step_name = getattr(step, "__name__", str(step))
        result = step(ctx)
        ctx.prior[step_name] = result

        if result.terminal:
            result.metadata = {
                "skill": skill.name, "resolved_by": step_name, **result.metadata
            }
            return result

    if not skill.steps:
        return StepResult(
            value=None, metadata={"skill": skill.name, "resolved_by": "(empty)"}
        )

    last = skill.steps[-1]
    last_name = getattr(last, "__name__", str(last))
    result = ctx.prior[last_name]
    result.metadata = {
        "skill": skill.name, "resolved_by": last_name, **result.metadata
    }
    return result

def compose(*skills: Skill) -> str:
    """Concatenate system prompts from multiple skills into one string."""
    return "\n\n".join(s.system_prompt for s in skills if s.system_prompt)

_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*\n?(.*?)\n?\s*```$",
    re.DOTALL,
)

def strip_fences(text: str) -> str:
    """Remove Markdown code fences."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text

__all__ = [
    "Caller",
    "Completion",
    "Skill",
    "Step",
    "StepContext",
    "StepResult",
    "compose",
    "run_skill",
    "strip_fences",
]
