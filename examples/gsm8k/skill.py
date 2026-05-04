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

import os
from typing import Any

from litellm import completion

from tk.llmbda import LMCaller, Skill, SkillContext, StepResult, lm, run_skill

REASON = "reason"
VERIFY = "verify"

MODEL = os.environ.get("GSM8K_MODEL", "openai/gpt-4o-mini")


def call_lm(*, messages: list[dict[str, str]], **kw: Any) -> str:
    """LiteLLM completion caller."""
    resp = completion(model=MODEL, messages=messages, **kw)
    return resp.choices[0].message.content


def scripted_gsm8k_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
    """Canned model for zero-dep demo runs."""
    text = messages[-1]["content"] if messages else ""
    if "verify" in text.lower() or "check" in text.lower():
        return "The solution is correct. The final answer is #### 42"
    return (
        "Step 1: We start with 20 apples.\n"
        "Step 2: We give away 5, leaving 20 - 5 = 15.\n"
        "Step 3: We buy 27 more, giving 15 + 27 = 42.\n"
        "#### 42"
    )


REASON_PROMPT = (
    "Solve the math problem step by step. Show your work clearly. "
    "End your response with #### followed by the final numeric answer."
)


@lm(call_lm, system_prompt=REASON_PROMPT)
def reason(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Chain-of-thought reasoning on the problem."""
    return StepResult(value=call(messages=[{"role": "user", "content": ctx.entry}]))


VERIFY_PROMPT = (
    "You are verifying a math solution. Check each arithmetic step. "
    "If correct, restate the final answer. If wrong, redo the calculation. "
    "End your response with #### followed by the final numeric answer."
)


@lm(call_lm, system_prompt=VERIFY_PROMPT)
def verify(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Self-check: ask the model to verify its own reasoning."""
    reasoning = ctx.trace[REASON].value
    prompt = f"Problem: {ctx.entry}\n\nSolution to verify:\n{reasoning}"
    return StepResult(value=call(messages=[{"role": "user", "content": prompt}]))


gsm8k = Skill(
    name="gsm8k",
    steps=[
        Skill(REASON, fn=reason),
        Skill(VERIFY, fn=verify),
    ],
)

__all__ = [
    "MODEL",
    "REASON",
    "VERIFY",
    "call_lm",
    "gsm8k",
    "run_skill",
    "scripted_gsm8k_model",
]
