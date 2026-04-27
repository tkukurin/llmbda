"""Tests for check_skill static graph validation."""

from __future__ import annotations

from tk.llmbda import (
    Skill,
    SkillContext,
    StepResult,
    check_skill,
)


def test_clean_linear_skill():
    def first(_ctx: SkillContext) -> StepResult:
        return StepResult(value=1, resolved=False)

    def second(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["first"].value + 1)

    skill = Skill(
        name="ok",
        steps=[Skill("first", fn=first), Skill("second", fn=second)],
    )
    assert check_skill(skill) == []


def test_catches_unknown_prior_key():
    def bad(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["nonexistent"].value)

    skill = Skill(name="bad", steps=[Skill("only", fn=bad)])
    issues = check_skill(skill)
    assert len(issues) == 1
    assert "nonexistent" in issues[0]
    assert "'only'" in issues[0]


def test_catches_unknown_prior_get_key():
    def bad(ctx: SkillContext) -> StepResult:
        prev = ctx.trace.get("typo")
        return StepResult(value=prev)

    skill = Skill(name="bad", steps=[Skill("step_a", fn=bad)])
    issues = check_skill(skill)
    assert len(issues) == 1
    assert "typo" in issues[0]


def test_step_cannot_reference_later_step():
    def first(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["second"].value, resolved=False)

    def second(_ctx: SkillContext) -> StepResult:
        return StepResult(value=2)

    skill = Skill(
        name="s",
        steps=[Skill("first", fn=first), Skill("second", fn=second)],
    )
    issues = check_skill(skill)
    assert any("second" in i for i in issues)


def test_nested_composite_sees_prior():
    def first(_ctx: SkillContext) -> StepResult:
        return StepResult(value=1)

    def second(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["first"].value + 1)

    skill = Skill(
        name="s",
        steps=[
            Skill(
                "group", steps=[Skill("first", fn=first), Skill("second", fn=second)]
            ),
        ],
    )
    assert check_skill(skill) == []


def test_nested_composite_later_group_sees_earlier():
    def a(_ctx: SkillContext) -> StepResult:
        return StepResult(value=1)

    def b(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["a"].value)

    skill = Skill(
        name="s",
        steps=[
            Skill("g1", steps=[Skill("a", fn=a)]),
            Skill("g2", steps=[Skill("b", fn=b)]),
        ],
    )
    assert check_skill(skill) == []


def test_unicode_step_names():
    def first(_ctx: SkillContext) -> StepResult:
        return StepResult(value=1, resolved=False)

    def second(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["λ::first"].value)

    skill = Skill(
        name="s",
        steps=[Skill("λ::first", fn=first), Skill("second", fn=second)],
    )
    assert check_skill(skill) == []


def test_unicode_step_name_typo():
    def bad(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["λ::extract"].value)

    extract = lambda _: StepResult(value=1, resolved=False)  # noqa: E731
    skill = Skill(
        name="s",
        steps=[Skill("ψ::extract", fn=extract), Skill("bad", fn=bad)],
    )
    issues = check_skill(skill)
    assert any("λ::extract" in i for i in issues)


def test_empty_skill():
    skill = Skill(name="empty", steps=[])
    assert check_skill(skill) == []


def test_multiple_issues_reported():
    def bad(ctx: SkillContext) -> StepResult:
        a = ctx.trace["ghost_a"]
        b = ctx.trace.get("ghost_b")
        return StepResult(value=(a, b))

    skill = Skill(name="s", steps=[Skill("bad", fn=bad)])
    issues = check_skill(skill)
    keys = set(issues)
    assert len(issues) == 2
    assert any("ghost_a" in i for i in keys)
    assert any("ghost_b" in i for i in keys)


def test_orchestrator_children_validated_as_separate_scope():
    """check_skill recurses into fn+steps children independently."""

    def orchestrator(_ctx: SkillContext) -> StepResult:
        return StepResult(value="ok")

    def bad_child(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["nonexistent"].value)

    skill = Skill(
        name="orch",
        fn=orchestrator,
        steps=[Skill("bad", fn=bad_child)],
    )
    issues = check_skill(Skill(name="s", steps=[skill]))
    assert any("nonexistent" in i for i in issues)


def test_orchestrator_children_valid_refs_pass():
    """Children that reference each other correctly produce no issues."""

    def orchestrator(_ctx: SkillContext) -> StepResult:
        return StepResult(value="ok")

    def first(_ctx: SkillContext) -> StepResult:
        return StepResult(value=1)

    def second(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["first"].value)

    skill = Skill(
        name="orch",
        fn=orchestrator,
        steps=[Skill("first", fn=first), Skill("second", fn=second)],
    )
    assert check_skill(Skill(name="s", steps=[skill])) == []


def test_orchestrator_children_cannot_see_outer_trace():
    """Children run in a fresh scope; outer step names aren't available."""

    def outer(_ctx: SkillContext) -> StepResult:
        return StepResult(value=1)

    def orchestrator(_ctx: SkillContext) -> StepResult:
        return StepResult(value="ok")

    def child_refs_outer(ctx: SkillContext) -> StepResult:
        return StepResult(value=ctx.trace["outer"].value)

    skill = Skill(
        name="s",
        steps=[
            Skill("outer", fn=outer),
            Skill(
                "orch",
                fn=orchestrator,
                steps=[Skill("child", fn=child_refs_outer)],
            ),
        ],
    )
    issues = check_skill(skill)
    assert any("outer" in i for i in issues)
