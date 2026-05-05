# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm>=1.0",
#     "tk-llmbda",
# ]
#
# [tool.uv.sources]
# tk-llmbda = { path = "../../", editable = true }
# ///
"""GSM8K solver: chain-of-thought reasoning with self-verification."""
from __future__ import annotations

from typing import Any

from litellm import completion

from tk.llmbda import (
    LMCaller,
    Skill,
    SkillContext,
    StepResult,
    last,
    lm,
    run_skill,
)

REASON = "ψ::reason"
VERIFY = "ψ::verify"

REASON_PROMPT = (
    "Solve the math problem step by step. Show your work clearly. "
    "End your response with #### followed by the final numeric answer."
)

VERIFY_PROMPT = (
    "You are verifying a math solution. Check each arithmetic step. "
    "If correct, restate the final answer. If wrong, redo the calculation. "
    "End your response with #### followed by the final numeric answer."
)


def make_skill(model: str) -> Skill:
    """Build gsm8k skill bound to the given model."""
    def call_lm(*, messages: list[dict[str, str]], **kw: Any) -> str:
        resp = completion(model=model, messages=messages, **kw)
        return resp.choices[0].message.content

    @lm(call_lm, system_prompt=REASON_PROMPT)
    def reason(ctx: SkillContext, call: LMCaller) -> StepResult:
        return StepResult(value=call(messages=[{"role": "user", "content": ctx.entry}]))

    @lm(call_lm, system_prompt=VERIFY_PROMPT)
    def verify(ctx: SkillContext, call: LMCaller) -> StepResult:
        reasoning = ctx.trace[REASON].value
        prompt = f"Problem: {ctx.entry}\n\nSolution to verify:\n{reasoning}"
        return StepResult(value=call(messages=[{"role": "user", "content": prompt}]))

    return Skill(
        name="gsm8k",
        steps=[Skill(REASON, fn=reason), Skill(VERIFY, fn=verify)],
    )


def runxp(model: str):
    """Run gsm8k skill with the given model."""
    trace = run_skill(make_skill(model), "What is 24 * 3 + 7?")
    print(last(trace).value)


__all__ = ["REASON", "VERIFY", "make_skill", "runxp"]
