"""tk.llmbda — skill composition for LLMs."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import wraps
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Any, Protocol

from tk.llmbda._check import check_skill

try:
    __version__ = _package_version("tk-llmbda")
except PackageNotFoundError:
    __version__ = "0+unknown"


class LMCaller(Protocol):
    """OpenAI-shape caller: keyword-only messages, arbitrary kwargs."""

    def __call__(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


@dataclass
class StepResult:
    """Skill output. Truthy `exits` short-circuits the remaining sequence."""

    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    exits: tuple[str, ...] | bool = ()

    def __post_init__(self) -> None:
        if not isinstance(self.exits, tuple):
            self.exits = ("self",) if self.exits else ()


ROOT = StepResult(value=None)  # sentinel


class _Trace(dict):
    """dict with informative KeyError for skill lookups."""

    def __missing__(self, key: str):
        available = ", ".join(self) or "(none)"
        raise KeyError(f"{key!r} not in trace (available: {available})")


@dataclass
class SkillContext:
    """Runtime state threaded through the skill sequence."""

    entry: Any
    trace: dict[str, StepResult] = field(default_factory=_Trace)
    prev: StepResult = field(default_factory=lambda: ROOT)


SkillFn = Callable[[SkillContext], StepResult]
OrchestratorFn = Callable[[SkillContext, list["Skill"]], StepResult]
LMSkillFn = Callable[[SkillContext, LMCaller], StepResult]


@dataclass
class Skill:
    """Recursive composition primitive.
    fn only: leaf step executed directly.
    steps only: composite; runtime walks children via DFS.
    fn + steps: fn is the orchestrator; children are declared for
        introspection and static checks but fn controls execution.
    """

    name: str
    fn: Callable[..., StepResult] | None = None
    steps: list[Skill] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        self.steps = [
            s
            if isinstance(s, Skill)
            else Skill(name=getattr(s, "__name__", str(s)), fn=s)
            for s in self.steps
        ]
        if not self.description and self.fn:
            self.description = inspect.getdoc(self.fn) or ""


@dataclass
class SkillResult:
    """Skill output with per-step trace."""

    skill: str
    resolved_by: tuple[str, ...]
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, StepResult] = field(default_factory=dict)


def lm(
    model: LMCaller,
    *,
    system_prompt: str = "",
) -> Callable[[Callable[..., StepResult]], Callable[..., StepResult]]:
    """Bind a skill fn to *model*; decorated fn is (ctx, call) or (ctx, steps, call)."""
    if system_prompt:

        def bound(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
            return model(
                messages=[{"role": "system", "content": system_prompt}, *messages],
                **kwargs,
            )
    else:
        bound = model

    def decorator(fn: Callable[..., StepResult]) -> Callable[..., StepResult]:
        @wraps(fn)
        def wrapper(ctx: SkillContext, *args: Any) -> StepResult:
            return fn(ctx, *args, bound)

        wrapper.lm_system_prompt = system_prompt  # type: ignore[attr-defined]
        wrapper.lm_model = model  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _trace_names(s: Skill) -> list[str]:
    return [s.name] if s.fn else [n for c in s.steps for n in _trace_names(c)]


def _dups(names: list[str]) -> list[str]:
    return [n for n, c in Counter(names).items() if c > 1]


def _child_scope_dups(s: Skill) -> list[str]:
    return [
        d
        for c in s.steps
        for d in (
            _dup_trace_names(Skill(name=c.name, steps=c.steps))
            if c.fn and c.steps
            else _child_scope_dups(c)
        )
    ]


def _dup_trace_names(skill: Skill) -> list[str]:
    inner = Skill(name=skill.name, steps=skill.steps)
    cs = _dups(_trace_names(inner)) if skill.fn and skill.steps else []
    return [*_dups(_trace_names(skill)), *cs, *_child_scope_dups(skill)]


def _walk(skill: Skill, ctx: SkillContext):
    """DFS walk yielding (name, result). Stops on truthy exits."""
    if skill.fn:
        result = skill.fn(ctx, skill.steps) if skill.steps else skill.fn(ctx)
        if not isinstance(result, StepResult):
            result = StepResult(value=result)
        ctx.trace[skill.name] = ctx.prev = result
        exited = bool(result.exits)
        yield skill.name, result
        return exited
    for child in skill.steps:
        if (yield from _walk(child, ctx)):
            return True
    return False


def _make_entry(entry: Any, kwargs: dict[str, Any]) -> Any:
    if kwargs and entry is not None:
        msg = "pass either positional entry or kwargs, not both"
        raise TypeError(msg)
    return kwargs or entry


def iter_skill(
    skill: Skill,
    entry: Any = None,
    **kwargs: Any,
) -> Iterator[tuple[str, StepResult]]:
    """Yield (name, result) per executed skill. Stops on truthy exits or after last."""
    if duplicates := _dup_trace_names(skill):
        raise ValueError(duplicates)
    ctx = SkillContext(entry=_make_entry(entry, kwargs))
    yield from _walk(skill, ctx)


def run_skill(skill: Skill, entry: Any = None, **kwargs: Any) -> SkillResult:
    """Run *skill* to completion."""
    if not (trace := dict(iter_skill(skill, _make_entry(entry, kwargs)))):
        return SkillResult(skill=skill.name, resolved_by=(), value=None)
    last_name, last = next(reversed(trace.items()))
    trail = last.exits[:-1] if last.exits[-1:] == ("self",) else last.exits
    return SkillResult(
        skill=skill.name,
        resolved_by=(last_name, *trail),
        value=last.value,
        metadata=last.metadata,
        trace=trace,
    )


_FENCE = "```"


def strip_fences(text: str) -> str:
    """Strip surrounding markdown code fences, if any."""
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
    "ROOT",
    "LMCaller",
    "LMSkillFn",
    "OrchestratorFn",
    "Skill",
    "SkillContext",
    "SkillFn",
    "SkillResult",
    "StepResult",
    "__version__",
    "check_skill",
    "iter_skill",
    "lm",
    "run_skill",
    "strip_fences",
]
