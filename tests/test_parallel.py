from tk.llmbda import Skill, SkillContext, StepResult


def test_parallel_steps_inherit_trace(run):
    def step_a(_ctx: SkillContext) -> StepResult:
        return StepResult(value="A")

    def step_b(ctx: SkillContext) -> StepResult:
        assert "a" in ctx.trace
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
