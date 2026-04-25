"""Skill execution and composition.

:func:`run_skill` executes a skill's steps in order.
:func:`compose` concatenates skill prompts for multi-skill pipelines.
"""

from __future__ import annotations

from typing import Any

from .skill import Caller, Skill, StepContext, StepResult


def run_skill(
    skill: Skill,
    entry: dict[str, Any],
    caller: Caller | None = None,
) -> dict[str, Any]:
    """Execute *skill*'s steps in order and return ``{value, metadata}``.

    The first step whose :attr:`StepResult.terminal` is ``True``
    becomes the final answer.  If no step is terminal the last
    step's result is used.  An empty skill returns ``None``.
    """
    ctx = StepContext(
        entry=entry,
        caller=caller,
        system_prompt=skill.system_prompt,
    )

    for step in skill.steps:
        result = step.run(ctx)
        ctx.prior[step.name] = result

        if result.terminal:
            return _wrap(skill.name, step.name, result)

    if not skill.steps:  # no explicit terminal step -> return last
        return _wrap(skill.name, "(empty)", StepResult())

    last = skill.steps[-1]
    return _wrap(skill.name, last.name, ctx.prior[last.name])


def compose(*skills: Skill) -> str:
    """Concatenate system prompts from multiple skills into one string."""
    return "\n\n".join(s.system_prompt for s in skills if s.system_prompt)


def _wrap(
    skill_name: str, step_name: str, result: StepResult,
) -> dict[str, Any]:
    return {
        "value": result.value,
        "metadata": {
            "skill": skill_name,
            "resolved_by": step_name,
            **result.metadata,
        },
    }
