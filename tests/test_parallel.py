from tk.llmbda import Skill, SkillContext, StepResult, run_skill, arun_skill
import pytest
import asyncio

def test_parallel_steps_inherit_trace(run):
    def step_a(ctx: SkillContext) -> StepResult:
        return StepResult(value="A")

    def step_b(ctx: SkillContext) -> StepResult:
        assert "a" in ctx.trace, "Trace from prior steps lost in parallel execution"
        assert ctx.trace["a"].value == "A"
        return StepResult(value="B")

    skill = Skill(
        name="root",
        steps=[
            Skill("a", fn=step_a),
            Skill("parallel_group", parallel=True, steps=[
                Skill("b", fn=step_b)
            ])
        ]
    )

    result = run(skill, "entry")
    assert result.trace["b"].value == "B"
