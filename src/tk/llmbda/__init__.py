"""tk.llmbda -- skill composition for LLMs.

Like lambda calculus, but the functions talk back.
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol


class LMCaller(Protocol):
    def __call__(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


@dataclass
class StepResult:
    """What a single step produces. ``resolved=False`` falls to the next step."""
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = True


@dataclass
class Step:
    """Named unit of work.
    description: human-readable summary; docstring fallback. Separate from
            @lm system prompts (read those via fn.lm_system_prompt).
    """
    name: str
    fn: StepFn
    description: str = ""
    def __post_init__(self) -> None:
        if not self.description:
            self.description = inspect.getdoc(self.fn) or ""


@dataclass
class StepContext:
    """Accumulator threaded through the step sequence."""
    entry: Any
    steps: list[Step] = field(default_factory=list)
    prior: dict[str, StepResult] = field(default_factory=dict)


StepFn = Callable[[StepContext], StepResult]
LMStepFn = Callable[[StepContext, LMCaller], StepResult]


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


def lm(
    model: LMCaller, *, system_prompt: str = "",
) -> Callable[[LMStepFn], StepFn]:
    """Bind a step function to a model, optionally prepending a system prompt.

    The decorated function receives ``(ctx, call)`` where ``call`` forwards to
    *model* with the system prompt (if any) prepended to ``messages``.
    """
    if system_prompt:
        def bound(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
            return model(
                messages=[{"role": "system", "content": system_prompt}, *messages],
                **kwargs,
            )
    else:
        bound = model
    def decorator(fn: LMStepFn) -> StepFn:
        @wraps(fn)
        def wrapper(ctx: StepContext) -> StepResult:
            return fn(ctx, bound)
        wrapper.lm_system_prompt = system_prompt  # type: ignore[attr-defined]
        wrapper.lm_model = model  # type: ignore[attr-defined]
        return wrapper
    return decorator


def iter_skill(skill: Skill, entry: Any) -> Iterator[tuple[str, StepResult]]:
    """Yield ``(step_name, result)`` for each executed step, in order.

    Execution stops when a step resolves or after the final step.
    """
    steps = list(skill.steps)
    if dups := [k for k, n in Counter(s.name for s in steps).items() if n > 1]:
        raise ValueError(dups)

    ctx = StepContext(entry=entry, steps=steps)
    last_idx = len(steps) - 1
    for i, step in enumerate(steps):
        result = step.fn(ctx)
        ctx.prior[step.name] = result
        should_stop = result.resolved or i == last_idx
        yield step.name, result
        if should_stop:
            return


def run_skill(skill: Skill, entry: Any) -> SkillResult:
    """Execute *skill* to completion and return the final ``SkillResult``."""
    trace: dict[str, StepResult] = {}
    last: StepResult | None = None
    last_name = "(empty)"
    for name, result in iter_skill(skill, entry):
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


_FENCE = "```"


def strip_fences(text: str) -> str:
    """Remove Markdown code fences from *text*, if any."""
    text = text.strip()
    if not text.startswith(_FENCE) or not text.endswith(_FENCE):
        return text
    lines = text.splitlines()
    match lines:
        case [opening, *body, closing] if (
            opening.startswith(_FENCE) and closing.strip() == _FENCE
        ):
            return "\n".join(body).strip()
    return text.removeprefix(_FENCE).removesuffix(_FENCE).strip()


__all__ = [
    "LMCaller",
    "LMStepFn",
    "Skill",
    "SkillResult",
    "Step",
    "StepContext",
    "StepFn",
    "StepResult",
    "iter_skill",
    "lm",
    "run_skill",
    "strip_fences",
]
