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
"""CRAG skill: 2-step elicitation (RELEVANCE retrieval, then generate answer)."""
from __future__ import annotations

import json
from typing import Any

from litellm import completion

from tk.llmbda import LMCaller, Skill, SkillContext, StepResult, lm, last, run_skill, strip_fences

RELEVANCE = "ψ::relevance"
GENERATE = "ψ::generate"

RELEVANCE_PROMPT = """\
You are a retrieval evaluator. Given a question and a list of documents,
judge which documents are relevant to answering the question.
Return ONLY a JSON array of objects: [{"title": "...", "relevant": true/false}, ...]"""

GENERATE_PROMPT = """\
You are a QA system. Answer the question using ONLY the provided documents.
Return ONLY JSON: {"answer": "<short factoid answer>"}"""


def make_skill(model: str) -> Skill:
    """Build crag skill bound to the given model."""
    def call_lm(*, messages: list[dict[str, str]], **kw: Any) -> str:
        resp = completion(model=model, messages=messages, **kw)
        return resp.choices[0].message.content

    @lm(call_lm, system_prompt=RELEVANCE_PROMPT)
    def relevance(ctx: SkillContext, call: LMCaller) -> StepResult:
        q, docs = ctx.entry["question"], ctx.entry["documents"]
        payload = json.dumps({"question": q, "documents": docs})
        raw = call(messages=[{"role": "user", "content": payload}])
        evaluations = json.loads(strip_fences(raw))
        relevant_docs = [
            doc
            for doc, ev in zip(docs, evaluations, strict=False)
            if ev.get("relevant", False)
        ]
        return StepResult(
            value={"relevant": relevant_docs, "evaluations": evaluations},
            meta={"n_relevant": len(relevant_docs), "n_total": len(docs)},
        )

    @lm(call_lm, system_prompt=GENERATE_PROMPT)
    def generate(ctx: SkillContext, call: LMCaller) -> StepResult:
        relevant_docs = ctx.trace[RELEVANCE].value["relevant"]
        question = ctx.entry["question"]
        payload = json.dumps({"question": question, "documents": relevant_docs})
        raw = call(messages=[{"role": "user", "content": payload}])
        parsed = json.loads(strip_fences(raw))
        answer = parsed.get("answer", "").strip()
        return StepResult(
            value={"answer": answer},
            meta={"n_docs_used": len(relevant_docs)},
        )

    return Skill(name="crag", steps=[Skill(RELEVANCE, fn=relevance), Skill(GENERATE, fn=generate)])


def runxp(model: str):
    """Run crag skill with the given model."""
    entry = {
        "question": "What is the capital of France?",
        "documents": [
            {"title": "France", "text": "The capital of France is Paris."},
            {"title": "Germany", "text": "The capital of Germany is Berlin."},
        ],
    }
    trace = run_skill(make_skill(model), entry)
    print(last(trace).value)


__all__ = ["GENERATE", "RELEVANCE", "make_skill", "runxp"]
