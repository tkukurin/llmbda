"""tk.llmbda — skill composition for LLMs."""

from __future__ import annotations

import asyncio
import inspect
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
    """Skill output. resolved=True short-circuits the remaining sequence."""
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_by: tuple[str, ...] = ()


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
    parallel: bool = False
    def __post_init__(self) -> None:
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


def _trace_names(skill: Skill) -> list[str]:
    """Names written to the current ctx.trace scope."""
    if skill.fn:
        return [skill.name]
    return [name for child in skill.steps for name in _trace_names(child)]


def _duplicate_names(names: list[str]) -> list[str]:
    """Names that appear more than once."""
    return [name for name, n in Counter(names).items() if n > 1]


def _child_scope_duplicates(skill: Skill) -> list[str]:
    """Duplicate names in orchestrator child trace scopes."""
    return [
        duplicate
        for child in skill.steps
        for duplicate in (
            _duplicate_trace_names(Skill(name=child.name, steps=child.steps))
            if child.fn and child.steps
            else _child_scope_duplicates(child)
        )
    ]


def _duplicate_trace_names(skill: Skill) -> list[str]:
    """Duplicate names per trace scope."""
    child_scope = (
        _duplicate_names(_trace_names(Skill(name=skill.name, steps=skill.steps)))
        if skill.fn and skill.steps
        else []
    )
    return [
        *_duplicate_names(_trace_names(skill)),
        *child_scope,
        *_child_scope_duplicates(skill),
    ]


async def _awalk(skill: Skill, ctx: SkillContext):
    """Async DFS walk yielding (name, str | StepResult)."""
    if skill.fn:
        coro = skill.fn(ctx, skill.steps) if skill.steps else skill.fn(ctx)
        if inspect.isasyncgen(coro):
            result = None
            async for item in coro:
                if isinstance(item, StepResult):
                    result = item
                else:
                    yield skill.name, item
            if result is None:
                msg = f"{skill.name}: streaming step must yield a final StepResult"
                raise TypeError(msg)
        elif inspect.isawaitable(coro):
            result = await coro
        else:
            result = coro
        ctx.trace[skill.name] = result
        ctx.prev = result
        yield skill.name, result
        return
    if skill.parallel:
        queue: asyncio.Queue[tuple[str, str | StepResult] | None] = asyncio.Queue()
        base_trace, base_prev = _Trace(ctx.trace), ctx.prev

        async def _run_child(child: Skill) -> None:
            child_ctx = SkillContext(
                entry=ctx.entry, trace=_Trace(base_trace), prev=base_prev,
            )
            try:
                async for name, item in _awalk(child, child_ctx):
                    await queue.put((name, item))
            finally:
                await queue.put(None)

        tasks = [asyncio.create_task(_run_child(c)) for c in skill.steps]
        active_tasks = len(tasks)

        while active_tasks > 0:
            res = await queue.get()
            if res is None:
                active_tasks -= 1
            else:
                name, item = res
                if isinstance(item, StepResult):
                    ctx.trace[name] = item
                    ctx.prev = item
                yield name, item

        await asyncio.gather(*tasks)
        return
    for child in skill.steps:
        async for name, item in _awalk(child, ctx):
            yield name, item
            if isinstance(item, StepResult) and item.resolved:
                return


async def aiter_skill(
    skill: Skill, entry: Any,
) -> AsyncIterator[tuple[str, str | StepResult]]:
    """Yield (name, str_chunk | StepResult) per executed skill (async)."""
    if duplicates := _duplicate_trace_names(skill):
        raise ValueError(duplicates)
    ctx = SkillContext(entry=entry)
    async for item in _awalk(skill, ctx):
        yield item


async def arun_skill(skill: Skill, entry: Any) -> SkillResult:
    """Run *skill* to completion (async)."""
    trace: dict[str, StepResult] = {}
    last: StepResult | None = None
    last_name = "(empty)"
    async for name, item in aiter_skill(skill, entry):
        if isinstance(item, str):
            continue
        trace[name] = item
        last, last_name = item, name
    if last is None:
        return SkillResult(skill=skill.name, resolved_by=(), value=None)
    return SkillResult(
        skill=skill.name,
        resolved_by=(last_name, *last.resolved_by),
        value=last.value,
        metadata=last.metadata,
        trace=trace,
    )


async def _drain_stream(agen) -> StepResult:
    """Consume an async generator, return the final StepResult."""
    final = None
    async for item in agen:
        if isinstance(item, StepResult):
            final = item
    if final is None:
        msg = "streaming step must yield a final StepResult"
        raise TypeError(msg)
    return final


def _resolve(result: object) -> StepResult:
    """Resolve sync, async, or async-generator step return values."""
    if inspect.isasyncgen(result):
        return asyncio.run(_drain_stream(result))
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result  # type: ignore[return-value]


def _walk(skill: Skill, ctx: SkillContext):
    """DFS walk yielding (name, result). Returns resolved bool."""
    if skill.fn:
        result = skill.fn(ctx, skill.steps) if skill.steps else skill.fn(ctx)
        if inspect.isgenerator(result):
            final = None
            for item in result:
                if isinstance(item, StepResult):
                    final = item
                else:
                    yield skill.name, item
            if final is None:
                msg = f"{skill.name}: streaming step must yield a final StepResult"
                raise TypeError(msg)
            result = final
        else:
            result = _resolve(result)
        ctx.trace[skill.name] = result
        ctx.prev = result
        resolved = result.resolved
        yield skill.name, result
        return resolved
    if skill.parallel:
        base_trace, base_prev = _Trace(ctx.trace), ctx.prev
        for child in skill.steps:
            child_ctx = SkillContext(
                entry=ctx.entry, trace=_Trace(base_trace), prev=base_prev,
            )
            for name, result in _walk(child, child_ctx):
                if isinstance(result, StepResult):
                    ctx.trace[name] = result
                    ctx.prev = result
                yield name, result
        return False
    for child in skill.steps:
        resolved = yield from _walk(child, ctx)
        if resolved:
            return True
    return False


def iter_skill(skill: Skill, entry: Any) -> Iterator[tuple[str, str | StepResult]]:
    """Yield (name, str_chunk | StepResult) per executed skill."""
    if duplicates := _duplicate_trace_names(skill):
        raise ValueError(duplicates)
    ctx = SkillContext(entry=entry)
    yield from _walk(skill, ctx)


def run_skill(skill: Skill, entry: Any) -> SkillResult:
    """Run *skill* to completion."""
    trace: dict[str, StepResult] = {}
    last: StepResult | None = None
    last_name = "(empty)"
    for name, item in iter_skill(skill, entry):
        if isinstance(item, str):
            continue
        trace[name] = item
        last, last_name = item, name
    if last is None:
        return SkillResult(skill=skill.name, resolved_by=(), value=None)
    return SkillResult(
        skill=skill.name,
        resolved_by=(last_name, *last.resolved_by),
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
    "aiter_skill",
    "arun_skill",
    "check_skill",
    "iter_skill",
    "lm",
    "run_skill",
    "strip_fences",
]
