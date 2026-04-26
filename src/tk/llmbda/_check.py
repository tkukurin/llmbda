"""Static validation of trace-key references via AST analysis."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tk.llmbda import Skill, SkillFn


def _leaves(skill: Skill) -> list[Skill]:
    """Leaf skills (those with fn) via DFS."""
    if skill.fn:
        return [skill]
    out: list[Skill] = []
    for child in skill.steps:
        out.extend(_leaves(child))
    return out


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
        if skill.steps:
            child_available: set[str] = set()
            for child in skill.steps:
                _check(child, child_available, issues)
                child_available.update(s.name for s in _leaves(child))
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
