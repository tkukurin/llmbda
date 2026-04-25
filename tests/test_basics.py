"""Smoke tests for tk.llmbda core."""

from typing import Any
from tk.llmbda import Skill, StepContext, StepResult, compose, run_skill


def echo_step(ctx: StepContext) -> StepResult:
    """Step that just returns the entry as-is."""
    return StepResult(value=ctx.entry, terminal=True)


def counting_step(ctx: StepContext) -> StepResult:
    """Step that counts prior steps."""
    return StepResult(
        value=len(ctx.prior),
        metadata={"seen": list(ctx.prior.keys())},
    )


class TestRunSkill:
    def test_single_step(self):
        skill = Skill(name="echo", steps=[echo_step])
        result = run_skill(skill, {"x": 1})
        assert result.value == {"x": 1}
        assert result.metadata["skill"] == "echo"
        assert result.metadata["resolved_by"] == "echo_step"

    def test_terminal_short_circuits(self):
        def _terminal(_ctx):
            return StepResult(value="stopped", terminal=True)

        def _unreachable(_ctx):
            msg = "should not be called"
            raise AssertionError(msg)

        skill = Skill(name="short", steps=[_terminal, _unreachable])
        result = run_skill(skill, {})
        assert result.value == "stopped"

    def test_last_step_wins_when_none_terminal(self):
        def a(ctx): return counting_step(ctx)
        def b(ctx): return counting_step(ctx)
        def c(ctx): return counting_step(ctx)

        skill = Skill(name="chain", steps=[a, b, c])
        result = run_skill(skill, {})
        assert result.value == 2  # step c sees a and b
        assert result.metadata["resolved_by"] == "c"

    def test_prior_accumulates(self):
        def deposit(_ctx):
            return StepResult(value="first", metadata={"order": 1})

        def check(ctx):
            prior_val = ctx.prior["deposit"].value
            return StepResult(value=f"saw {prior_val}", terminal=True)

        skill = Skill(name="acc", steps=[deposit, check])
        result = run_skill(skill, {})
        assert result.value == "saw first"

    def test_empty_skill(self):
        skill = Skill(name="noop")
        result = run_skill(skill, {"x": 1})
        assert result.value is None
        assert result.metadata["skill"] == "noop"

    def test_metadata_merges(self):
        def _with_meta(_ctx):
            return StepResult(
                value=42,
                metadata={"custom": "data", "extra": True},
                terminal=True,
            )

        skill = Skill(name="meta", steps=[_with_meta])
        result = run_skill(skill, {})
        assert result.metadata["custom"] == "data"
        assert result.metadata["extra"] is True
        assert result.metadata["skill"] == "meta"


class TestCompose:
    def test_concatenates_prompts(self):
        s1 = Skill(name="a", system_prompt="You are helpful.")
        s2 = Skill(name="b", system_prompt="Be concise.")
        assert compose(s1, s2) == "You are helpful.\n\nBe concise."

    def test_skips_empty_prompts(self):
        s1 = Skill(name="a", system_prompt="Only this.")
        s2 = Skill(name="b")
        assert compose(s1, s2) == "Only this."

    def test_empty_compose(self):
        assert compose() == ""


class DummyResponse:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


class TestCallerIntegration:
    def test_caller_available_in_context(self):
        recorded = []

        def spy_step(ctx):
            recorded.append(ctx.caller)
            return StepResult(value="ok", terminal=True)

        def fake_caller(**_kw) -> Any:
            return DummyResponse('{"x": 1}')

        skill = Skill(name="spy", steps=[spy_step])
        run_skill(skill, {}, caller=fake_caller)
        assert recorded == [fake_caller]

    def test_step_can_call_caller(self):
        def llm_step(ctx):
            raw = ctx.caller(messages=[{"role": "user", "content": "hi"}])
            return StepResult(value=raw.choices[0].message.content, terminal=True)

        def fake_caller(**_kw) -> Any:
            return DummyResponse("hello back")

        skill = Skill(name="chat", steps=[llm_step])
        result = run_skill(skill, {}, caller=fake_caller)
        assert result.value == "hello back"
