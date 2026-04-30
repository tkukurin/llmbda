"""tk.llmbda — skill composition for LLMs."""

from __future__ import annotations

import asyncio
import inspect as _stdlib_inspect
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator
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
    """Step output: a value and optional auxiliary metadata."""

    value: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


Trace = dict[str, StepResult]


class _Trace(dict):
    """dict with informative KeyError for skill lookups."""

    def __missing__(self, key: str):
        available = ", ".join(self) or "(none)"
        raise KeyError(f"{key!r} not in trace (available: {available})")


@dataclass
class SkillContext:
    """Runtime state threaded through the skill sequence."""

    entry: Any
    trace: Trace = field(default_factory=_Trace)
    prev: StepResult = field(default_factory=StepResult)


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
            self.description = _stdlib_inspect.getdoc(self.fn) or ""


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
        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(ctx: SkillContext, *args: Any) -> StepResult:
                return await fn(ctx, *args, bound)

            async_wrapper.lm_system_prompt = system_prompt  # type: ignore[attr-defined]
            async_wrapper.lm_model = model  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]

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


def _walk(skill: Skill, ctx: SkillContext) -> Iterator[tuple[str, StepResult]]:
    """DFS walk yielding (name, result)."""
    if skill.fn:
        result = skill.fn(ctx, skill.steps) if skill.steps else skill.fn(ctx)
        if not isinstance(result, StepResult):
            result = StepResult(value=result)
        ctx.trace[skill.name] = ctx.prev = result
        yield skill.name, result
    else:
        for child in skill.steps:
            yield from _walk(child, ctx)


async def _awalk(
    skill: Skill, ctx: SkillContext
) -> AsyncIterator[tuple[str, StepResult]]:
    """Async DFS walk; awaits coroutine fns, calls sync fns inline."""
    if skill.fn:
        if asyncio.iscoroutinefunction(skill.fn):
            result = (
                await skill.fn(ctx, skill.steps)
                if skill.steps
                else await skill.fn(ctx)
            )
        else:
            result = skill.fn(ctx, skill.steps) if skill.steps else skill.fn(ctx)
        if not isinstance(result, StepResult):
            result = StepResult(value=result)
        ctx.trace[skill.name] = ctx.prev = result
        yield skill.name, result
    else:
        for child in skill.steps:
            async for pair in _awalk(child, ctx):
                yield pair


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
    """Yield (name, result) per executed step."""
    if duplicates := _dup_trace_names(skill):
        raise ValueError(duplicates)
    ctx = SkillContext(entry=_make_entry(entry, kwargs))
    yield from _walk(skill, ctx)


def run_skill(skill: Skill, entry: Any = None, **kwargs: Any) -> Trace:
    """Run *skill* to completion, return ordered trace dict."""
    return dict(iter_skill(skill, _make_entry(entry, kwargs)))


async def aiter_skill(
    skill: Skill,
    entry: Any = None,
    **kwargs: Any,
) -> AsyncIterator[tuple[str, StepResult]]:
    """Async yield (name, result) per executed step."""
    if duplicates := _dup_trace_names(skill):
        raise ValueError(duplicates)
    ctx = SkillContext(entry=_make_entry(entry, kwargs))
    async for pair in _awalk(skill, ctx):
        yield pair


async def arun_skill(skill: Skill, entry: Any = None, **kwargs: Any) -> Trace:
    """Async run *skill* to completion, return ordered trace dict."""
    return {
        name: result
        async for name, result in aiter_skill(skill, _make_entry(entry, kwargs))
    }


def last(trace: Trace) -> StepResult:
    """Last step's result from a trace."""
    if not trace:
        return StepResult()
    return trace[next(reversed(trace))]


def fst_match(ctx: SkillContext, steps: list[Skill]) -> StepResult:
    """Orchestrator: return the first child result with a non-None value."""
    for s in steps:
        r = run_skill(Skill(name="_", steps=[s]), ctx.entry)
        v = last(r)
        if v.value is not None:
            return v
    return StepResult()


async def afst_match(ctx: SkillContext, steps: list[Skill]) -> StepResult:
    """Async orchestrator: return first child result with a non-None value."""
    for s in steps:
        r = await arun_skill(Skill(name="_", steps=[s]), ctx.entry)
        v = last(r)
        if v.value is not None:
            return v
    return StepResult()


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
    "LMCaller",
    "LMSkillFn",
    "OrchestratorFn",
    "Skill",
    "SkillContext",
    "SkillFn",
    "StepResult",
    "Trace",
    "__version__",
    "afst_match",
    "aiter_skill",
    "arun_skill",
    "check_skill",
    "fst_match",
    "iter_skill",
    "last",
    "lm",
    "run_skill",
    "strip_fences",
]
