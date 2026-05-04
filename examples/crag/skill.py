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
"""CRAG skill: 2-step elicitation (evaluate retrieval, then generate answer)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from litellm import completion

from tk.llmbda import (
    LMCaller,
    Skill,
    SkillContext,
    StepResult,
    lm,
    run_skill,
    strip_fences,
)

EVALUATE = "ψ::evaluate"
GENERATE = "ψ::generate"

MODEL = os.environ.get("CRAG_MODEL", "openai/gpt-4o-mini")


def call_lm(*, messages: list[dict[str, str]], **kw: Any) -> str:
    resp = completion(model=MODEL, messages=messages, **kw)
    return resp.choices[0].message.content  # type: ignore[union-attr]


def scripted_crag_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
    """Zero-dependency scripted model for demo runs."""
    system = (
        messages[0]["content"].lower()
        if messages and messages[0]["role"] == "system"
        else ""
    )
    user_content = messages[-1]["content"]
    if "retrieval evaluator" in system:
        payload = json.loads(strip_fences(user_content))
        q_tokens = set(re.findall(r"\w+", payload["question"].lower()))
        results = []
        for doc in payload["documents"]:
            doc_tokens = set(re.findall(r"\w+", doc["text"].lower()))
            overlap = len(q_tokens & doc_tokens) / (len(q_tokens) or 1)
            label = "relevant" if overlap >= 0.3 else "irrelevant"
            results.append({"title": doc["title"], "relevant": label == "relevant"})
        return json.dumps(results)
    if "answer the question" in system:
        payload = json.loads(strip_fences(user_content))
        docs = payload.get("documents", [])
        context = " ".join(d["text"] for d in docs)
        q_tokens = set(re.findall(r"\w+", payload["question"].lower()))
        sents = [s.strip() for s in re.split(r"[.!?]+", context) if s.strip()]
        best, best_score = "unknown", 0.0
        for sent in sents:
            s_tokens = set(re.findall(r"\w+", sent.lower()))
            score = len(q_tokens & s_tokens) / (len(q_tokens) or 1)
            if score > best_score:
                best, best_score = sent, score
        answer = best if best_score > 0 else "unknown"
        if len(answer.split()) > 12:
            non_q = [w for w in answer.split() if w.lower() not in q_tokens]
            answer = " ".join(non_q[:6]) if non_q else " ".join(answer.split()[:6])
        return json.dumps({"answer": answer})
    msg = f"unknown scripted prompt: {system[:60]}"
    raise ValueError(msg)


EVALUATE_PROMPT = """\
You are a retrieval evaluator. Given a question and a list of documents,
judge which documents are relevant to answering the question.
Return ONLY a JSON array of objects: [{"title": "...", "relevant": true/false}, ...]"""

GENERATE_PROMPT = """\
You are a QA system. Answer the question using ONLY the provided documents.
Return ONLY JSON: {"answer": "<short factoid answer>"}"""


@lm(call_lm, system_prompt=EVALUATE_PROMPT)
def evaluate(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Ask the model which retrieved documents are relevant."""
    entry = ctx.entry
    payload = json.dumps(
        {"question": entry["question"], "documents": entry["documents"]}
    )
    raw = call(messages=[{"role": "user", "content": payload}])
    evaluations = json.loads(strip_fences(raw))
    relevant_docs = [
        doc
        for doc, ev in zip(entry["documents"], evaluations, strict=False)
        if ev.get("relevant", False)
    ]
    return StepResult(
        value={"relevant_documents": relevant_docs, "evaluations": evaluations},
        meta={"n_relevant": len(relevant_docs), "n_total": len(entry["documents"])},
    )


@lm(call_lm, system_prompt=GENERATE_PROMPT)
def generate(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Generate answer using only the relevant documents."""
    relevant_docs = ctx.trace[EVALUATE].value["relevant_documents"]
    question = ctx.entry["question"]
    payload = json.dumps({"question": question, "documents": relevant_docs})
    raw = call(messages=[{"role": "user", "content": payload}])
    parsed = json.loads(strip_fences(raw))
    answer = parsed.get("answer", "").strip()
    return StepResult(
        value={"answer": answer},
        meta={"n_docs_used": len(relevant_docs)},
    )


crag = Skill(
    name="crag",
    steps=[
        Skill(EVALUATE, fn=evaluate),
        Skill(GENERATE, fn=generate),
    ],
)

__all__ = [
    "EVALUATE",
    "GENERATE",
    "MODEL",
    "call_lm",
    "crag",
    "run_skill",
    "scripted_crag_model",
]
