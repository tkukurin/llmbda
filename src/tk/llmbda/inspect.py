"""Inspect AI adapters for `tk.llmbda` skills.

Note: this module is named `inspect` for user-facing symmetry with `inspect_ai`;
it does not shadow stdlib `inspect` because all imports here are absolute.
Requires the `inspect` extra: `pip install tk-llmbda[inspect]`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from tk.llmbda import Skill, run_skill

if TYPE_CHECKING:
    from collections.abc import Callable

    from inspect_ai.scorer import Score, Scorer
    from inspect_ai.solver import Solver, TaskState

    from tk.llmbda import StepResult

DEFAULT_TRACE_KEY = "llmbda.trace"


def skill_solver(
    skill: Skill,
    *,
    entry: Callable[[TaskState], Any] | None = None,
    trace_key: str = DEFAULT_TRACE_KEY,
) -> Solver:
    """Run *skill* once per sample; expose value as completion, trace in metadata."""
    from inspect_ai.model import ModelOutput  # noqa: PLC0415  (optional dep)
    from inspect_ai.solver import solver  # noqa: PLC0415

    extract = entry or (lambda s: s.input_text)

    @solver
    def _factory():
        async def solve(state, _generate):
            result = run_skill(skill, extract(state))
            state.output = ModelOutput.from_content(
                model=str(state.model),
                content="" if result.value is None else str(result.value),
            )
            if state.metadata is None:
                state.metadata = {}
            state.metadata[trace_key] = result.trace
            return state

        return solve

    return _factory()


def step_scorer(
    step_name: str,
    inner: Scorer,
    *,
    trace_key: str = DEFAULT_TRACE_KEY,
    project: Callable[[Any], str] = str,
    metrics: list | None = None,
) -> Scorer:
    """Apply *inner* scorer to the named step's value instead of final completion."""
    from inspect_ai.model import ModelOutput  # noqa: PLC0415  (optional dep)
    from inspect_ai.scorer import accuracy, scorer, stderr  # noqa: PLC0415

    resolved_metrics = metrics or _inherit_metrics(inner) or [accuracy(), stderr()]

    @scorer(metrics=resolved_metrics, name=f"step[{step_name}]")
    def _factory():
        async def score(state, target):
            trace = (state.metadata or {}).get(trace_key) or {}
            if step_name not in trace:
                available = ", ".join(trace) or "(none)"
                raise KeyError(f"{step_name!r} not in trace (available: {available})")
            step = trace[step_name]
            shadow = copy.copy(state)
            shadow.output = ModelOutput.from_content(
                model=str(state.model),
                content=project(step.value),
            )
            return await inner(shadow, target)

        return score

    return _factory()


def _inherit_metrics(inner: Scorer) -> list | None:
    """Best-effort read of metrics baked into *inner* at definition time."""
    try:
        from inspect_ai.scorer._scorer import scorer_metrics  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return list(scorer_metrics(inner)) or None
    except (AttributeError, KeyError, TypeError):
        return None


def step_check(
    step_name: str,
    predicate: Callable[[StepResult], float | bool | Score],
    *,
    trace_key: str = DEFAULT_TRACE_KEY,
    metrics: list | None = None,
) -> Scorer:
    """Score a step by predicate on its StepResult (value + metadata)."""
    from inspect_ai.scorer import Score as _Score  # noqa: PLC0415
    from inspect_ai.scorer import mean, scorer, stderr  # noqa: PLC0415

    resolved_metrics = metrics or [mean(), stderr()]

    @scorer(metrics=resolved_metrics, name=f"check[{step_name}]")
    def _factory():
        async def score(state, target):  # noqa: ARG001
            trace = (state.metadata or {}).get(trace_key) or {}
            if step_name not in trace:
                available = ", ".join(trace) or "(none)"
                raise KeyError(f"{step_name!r} not in trace (available: {available})")
            out = predicate(trace[step_name])
            if isinstance(out, _Score):
                return out
            val = float(out) if isinstance(out, (int, float)) else (1.0 if out else 0.0)
            return _Score(value=val)

        return score

    return _factory()


__all__ = ["DEFAULT_TRACE_KEY", "skill_solver", "step_check", "step_scorer"]
