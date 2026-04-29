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
"""GSM8K solver skill: CoT reasoning, extraction, verification, repair.

Plain module imported by `scoring.py`.
"""

from __future__ import annotations

import os
import re
from typing import Any

from litellm import completion

from tk.llmbda import LMCaller, Skill, SkillContext, StepResult, lm, run_skill

PARSE = "λ::parse"
REASON = "ψ::reason"
EXTRACT = "λ::extract"
VERIFY = "λ::verify"
REPAIR = "ψ::repair"

MODEL = os.environ.get("GSM8K_MODEL", "openai/gpt-4o-mini")

_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_CALC_RE = re.compile(r"<<([^>]+)=([^>]+)>>")
_NUMBER_RE = re.compile(r"-?[\d,]+\.?\d*")


def call_lm(*, messages: list[dict[str, str]], **kw: Any) -> str:
    resp = completion(model=MODEL, messages=messages, **kw)
    return resp.choices[0].message.content


def parse_problem(ctx: SkillContext) -> StepResult:
    """Normalize the input question into a clean string."""
    question = ctx.entry if isinstance(ctx.entry, str) else ctx.entry["question"]
    return StepResult({"question": question.strip()})


REASON_PROMPT = """\
You are a math tutor. Solve the problem step by step.
Use <<expression=result>> for each calculation.
End with #### followed by the final numeric answer on its own line."""


@lm(call_lm, system_prompt=REASON_PROMPT)
def reason(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Generate chain-of-thought reasoning via LLM."""
    question = ctx.trace[PARSE].value["question"]
    raw = call(messages=[{"role": "user", "content": question}])
    return StepResult({"reasoning": raw.strip()})


def extract_answer(ctx: SkillContext) -> StepResult:
    """Pull the final numeric answer from the CoT reasoning."""
    reasoning = ctx.trace[REASON].value["reasoning"]
    m = _ANSWER_RE.search(reasoning)
    if m:
        answer = m.group(1).strip().replace(",", "").replace("$", "")
    else:
        numbers = _NUMBER_RE.findall(reasoning)
        answer = numbers[-1].replace(",", "") if numbers else ""
    return StepResult(
        {"answer": answer, "reasoning": reasoning},
        {"extracted_from": "####" if m else "fallback"},
    )


def verify_arithmetic(ctx: SkillContext) -> StepResult:
    """Re-evaluate <<expr=result>> annotations; flag mismatches."""
    reasoning = ctx.trace[REASON].value["reasoning"]
    errors: list[str] = []
    for expr, claimed in _CALC_RE.findall(reasoning):
        claimed_clean = claimed.strip().replace(",", "")
        try:
            actual = _safe_eval(expr)
            if actual is not None and abs(actual - _parse_number(claimed_clean)) > 0.01:
                errors.append(f"{expr} = {claimed_clean} (expected {actual})")
        except (ValueError, ZeroDivisionError):
            pass
    prev = ctx.trace[EXTRACT].value
    return StepResult(
        {"answer": prev["answer"], "reasoning": reasoning},
        {"valid": len(errors) == 0, "errors": errors},
    )


def _safe_eval(expr: str) -> float | None:
    expr = expr.replace(",", "").strip()
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:  # noqa: BLE001
        return None


def _parse_number(s: str) -> float:
    return float(s.replace(",", "").replace("$", ""))


REPAIR_PROMPT = """\
You are a math tutor. The previous solution had arithmetic errors.
Fix ONLY the errors listed below, keep the same structure.
Use <<expression=result>> for each calculation.
End with #### followed by the corrected final numeric answer."""


@lm(call_lm, system_prompt=REPAIR_PROMPT)
def repair(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Re-prompt LLM to fix arithmetic errors found by verify step."""
    prev = ctx.prev
    if prev.metadata.get("valid", True):
        return StepResult(prev.value, {**prev.metadata, "repaired": False})
    question = ctx.trace[PARSE].value["question"]
    errors = prev.metadata["errors"]
    prompt = (
        f"Question: {question}\n\n"
        f"Previous reasoning:\n{prev.value['reasoning']}\n\n"
        f"Errors found:\n" + "\n".join(f"- {e}" for e in errors)
    )
    raw = call(messages=[{"role": "user", "content": prompt}])
    m = _ANSWER_RE.search(raw)
    answer = (
        m.group(1).strip().replace(",", "").replace("$", "")
        if m
        else prev.value["answer"]
    )
    return StepResult(
        {"answer": answer, "reasoning": raw.strip()},
        {"repaired": True, "original_errors": errors},
    )


gsm8k_solver = Skill(
    name="gsm8k_solver",
    steps=[
        Skill(PARSE, fn=parse_problem),
        Skill(REASON, fn=reason),
        Skill(EXTRACT, fn=extract_answer),
        Skill(VERIFY, fn=verify_arithmetic),
        Skill(REPAIR, fn=repair),
    ],
)

__all__ = [
    "EXTRACT",
    "MODEL",
    "PARSE",
    "REASON",
    "REPAIR",
    "VERIFY",
    "gsm8k_solver",
    "run_skill",
]
