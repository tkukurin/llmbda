import asyncio

import pytest

from tk.llmbda import Skill, SkillContext, StepResult, run_skill


async def async_step(_ctx: SkillContext) -> StepResult:
    return StepResult(value="A")


def test_sync_runner_raises_inside_event_loop():
    skill = Skill("a", fn=async_step)

    async def run_in_loop():
        run_skill(skill, "entry")

    with pytest.raises(RuntimeError, match="cannot be called"):
        asyncio.run(run_in_loop())
