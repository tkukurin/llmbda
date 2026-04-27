from tk.llmbda import Skill, SkillContext, StepResult, run_skill
import pytest
import asyncio

async def async_step(ctx: SkillContext) -> StepResult:
    return StepResult(value="A")

def test_async_step_runs_in_sync_runner_while_in_event_loop():
    skill = Skill("a", fn=async_step)
    
    async def run_in_loop():
        # This shouldn't raise RuntimeError anymore!
        result = run_skill(skill, "entry")
        return result.value
            
    assert asyncio.run(run_in_loop()) == "A"
