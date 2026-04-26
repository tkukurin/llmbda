"""tk.llmbda — skill composition for LLMs."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol

from tk.llmbda._version import __version__


class LMCaller(Protocol):
    """OpenAI-shape caller: keyword-only messages, arbitrary kwargs."""
    def __call__(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


@dataclass
class StepResult:
    """Skill output. resolved=True short-circuits the remaining sequence."""
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


ROOT = StepResult(value=None)  # sentinel


class _Trace(dict):
    """dict with informative KeyError for skill lookups."""
    def __missing__(self, key: str):
        available = ", ".join(self) or "(none)"
        raise KeyError(f"{key!r} not in trace (available: {available})")


@dataclass
class SkillContext:
    """Accumulator threaded through the skill sequence."""
    entry: Any
    skills: list[Skill] = field(default_factory=list)
    trace: dict[str, StepResult] = field(default_factory=_Trace)
    prev: StepResult = field(default_factory=lambda: ROOT)


SkillFn = Callable[[SkillContext], StepResult]
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
    fn: SkillFn | None = None
    steps: list[Skill] = field(default_factory=list)
    description: str = ""
    def __post_init__(self) -> None:
        if not self.description and self.fn:
            self.description = inspect.getdoc(self.fn) or ""


@dataclass
class SkillResult:
    """Skill output with per-step trace."""
    skill: str
    resolved_by: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, StepResult] = field(default_factory=dict)


def lm(
    model: LMCaller,
    *,
    system_prompt: str = "",
) -> Callable[[LMSkillFn], SkillFn]:
    """Bind a skill fn to *model*; decorated fn is (ctx, call)."""
    if system_prompt:
        def bound(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
            return model(
                messages=[{"role": "system", "content": system_prompt}, *messages],
                **kwargs,
            )
    else:
        bound = model

    def decorator(fn: LMSkillFn) -> SkillFn:
        @wraps(fn)
        def wrapper(ctx: SkillContext) -> StepResult:
            return fn(ctx, bound)
        wrapper.lm_system_prompt = system_prompt  # type: ignore[attr-defined]
        wrapper.lm_model = model  # type: ignore[attr-defined]
        return wrapper
    return decorator


def _leaves(skill: Skill) -> list[Skill]:
    """Leaf skills (those with fn) via DFS."""
    if skill.fn:
        return [skill]
    out: list[Skill] = []
    for child in skill.steps:
        out.extend(_leaves(child))
    return out


def _all_names(skill: Skill) -> list[str]:
    """All names written to ctx.trace by this skill tree."""
    if skill.fn:
        return [skill.name]
    names: list[str] = []
    for child in skill.steps:
        names.extend(_all_names(child))
    return names


def _walk(skill: Skill, ctx: SkillContext):
    """DFS walk yielding (name, result). Returns resolved bool."""
    if skill.fn:
        if skill.steps:
            prev_skills, ctx.skills = ctx.skills, skill.steps
        result = skill.fn(ctx)
        if skill.steps:
            ctx.skills = prev_skills
        ctx.trace[skill.name] = result
        ctx.prev = result
        resolved = result.resolved
        yield skill.name, result
        return resolved
    for child in skill.steps:
        resolved = yield from _walk(child, ctx)
        if resolved:
            return True
    return False


def iter_skill(skill: Skill, entry: Any) -> Iterator[tuple[str, StepResult]]:
    """Yield (name, result) per executed skill. Stops on resolved=True or after last."""
    names = _all_names(skill)
    if dups := [k for k, n in Counter(names).items() if n > 1]:
        raise ValueError(dups)
    ctx = SkillContext(entry=entry, skills=_leaves(skill))
    yield from _walk(skill, ctx)


def run_skill(skill: Skill, entry: Any) -> SkillResult:
    """Run *skill* to completion."""
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


def _prior_refs(fn: SkillFn) -> list[str]:
    """Extract string keys from ctx.trace[...] and ctx.trace.get(...) via AST."""
    target = getattr(fn, "__wrapped__", fn)
    try:
        source = textwrap.dedent(inspect.getsource(target))
        tree = ast.parse(source)
    except (OSError, TypeError):
        return []
    refs: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "trace"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            refs.append(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "trace"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            refs.append(node.args[0].value)
    return refs


def _check(skill: Skill, available: set[str], issues: list[str]) -> None:
    """Recursively validate trace references."""
    if skill.fn:
        issues.extend(
            f"'{skill.name}' references undeclared trace key '{ref}'"
            for ref in _prior_refs(skill.fn)
            if ref not in available
        )
        return
    current = set(available)
    for child in skill.steps:
        _check(child, current, issues)
        current.update(s.name for s in _leaves(child))


def check_skill(skill: Skill) -> list[str]:
    """Static validation: report trace key references that can't exist at runtime."""
    issues: list[str] = []
    _check(skill, set(), issues)
    return issues


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
