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
"""CRAG skill: Corrective Retrieval Augmented Generation.

Implements the pipeline from arXiv:2401.15884 (Yan et al., 2024):
  parse → retrieve → evaluate → action → refine → generate
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any

from tk.llmbda import (
    LMCaller,
    Skill,
    SkillContext,
    StepResult,
    lm,
    run_skill,
    strip_fences,
)

PARSE = "λ::parse"
RETRIEVE = "λ::retrieve"
EVALUATE = "ψ::evaluate"
ACTION = "λ::action"
REFINE = "λ::refine"
GENERATE = "ψ::generate"

MODEL = os.environ.get("CRAG_MODEL", "openai/gpt-4o-mini")
TOP_K = int(os.environ.get("CRAG_TOP_K", "5"))
CONFIDENCE_UPPER = float(os.environ.get("CRAG_CONF_UPPER", "0.7"))
CONFIDENCE_LOWER = float(os.environ.get("CRAG_CONF_LOWER", "0.3"))


_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n = len(corpus_tokens) or 1
    df: dict[str, int] = Counter()
    for doc in corpus_tokens:
        df.update(set(doc))
    return {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}


def _tfidf_score(
    query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]
) -> float:
    tf_doc = _tf(doc_tokens)
    return sum(tf_doc.get(t, 0.0) * idf.get(t, 0.0) for t in set(query_tokens))


def parse_query(ctx: SkillContext) -> StepResult:
    """Normalize HotpotQA record into question + paragraph list."""
    record = ctx.entry
    question = record["question"].strip()
    titles = record["context"]["title"]
    sentences_list = record["context"]["sentences"]
    paragraphs = []
    for title, sents in zip(titles, sentences_list, strict=False):
        text = " ".join(s.strip() for s in sents)
        paragraphs.append({"title": title, "text": text, "sentences": list(sents)})
    return StepResult(
        value={"question": question, "paragraphs": paragraphs},
        meta={"num_paragraphs": len(paragraphs)},
    )


def retrieve(ctx: SkillContext) -> StepResult:
    """Rank paragraphs by TF-IDF similarity to the query, return top-k."""
    parsed = ctx.trace[PARSE].value
    question = parsed["question"]
    paragraphs = parsed["paragraphs"]
    query_tokens = _tokenize(question)
    corpus_tokens = [_tokenize(p["text"]) for p in paragraphs]
    idf = _idf(corpus_tokens)
    scored = [
        (i, _tfidf_score(query_tokens, doc_tok, idf))
        for i, doc_tok in enumerate(corpus_tokens)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[:TOP_K]
    retrieved = [
        {**paragraphs[i], "score": score, "rank": rank}
        for rank, (i, score) in enumerate(top_k)
    ]
    return StepResult(
        value={"question": question, "documents": retrieved},
        meta={"scores": [s for _, s in top_k]},
    )


def _scripted_evaluate(question: str, documents: list[dict]) -> list[dict]:
    """Heuristic relevance evaluator for zero-dependency demo runs."""
    q_tokens = set(_tokenize(question))
    results = []
    for doc in documents:
        doc_tokens = set(_tokenize(doc["text"]))
        overlap = len(q_tokens & doc_tokens) / (len(q_tokens) or 1)
        if overlap >= 0.4:
            label, confidence = "correct", min(0.6 + overlap, 0.95)
        elif overlap >= 0.2:
            label, confidence = "ambiguous", 0.3 + overlap
        else:
            label, confidence = "incorrect", max(0.1, 0.3 - overlap)
        results.append(
            {
                "title": doc["title"],
                "label": label,
                "confidence": confidence,
            }
        )
    return results


def scripted_crag_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
    """Scripted LMCaller so this example runs without credentials."""
    system = (
        messages[0]["content"].lower()
        if messages and messages[0]["role"] == "system"
        else ""
    )
    payload = json.loads(strip_fences(messages[-1]["content"]))
    if "retrieval evaluator" in system:
        evals = _scripted_evaluate(payload["question"], payload["documents"])
        return json.dumps(evals)
    if "answer the question" in system:
        return _scripted_generate(payload)
    msg = f"unknown scripted prompt: {system[:60]}"
    raise ValueError(msg)


def _scripted_generate(payload: dict) -> str:
    """Heuristic answer generation: look for short factoid in context."""
    question = payload["question"].lower()
    context = payload.get("context", "")
    if not context:
        return json.dumps({"answer": "unknown", "confidence": 0.1})
    sents = [s.strip() for s in re.split(r"[.!?]+", context) if s.strip()]
    q_tokens = set(_tokenize(question))
    best_sent, best_score = "", 0.0
    for sent in sents:
        s_tokens = set(_tokenize(sent))
        overlap = len(q_tokens & s_tokens) / (len(q_tokens) or 1)
        if overlap > best_score:
            best_sent, best_score = sent, overlap
    answer = best_sent.strip() if best_sent else "unknown"
    if len(answer.split()) > 15:
        words = answer.split()
        non_q = [w for w in words if w.lower() not in q_tokens]
        answer = " ".join(non_q[:8]) if non_q else " ".join(words[:8])
    return json.dumps({"answer": answer, "confidence": round(best_score, 2)})


EVALUATE_PROMPT = """\
You are a retrieval evaluator. For each document, judge whether it is relevant
to answering the question. Return ONLY a JSON array of objects with keys:
title, label (one of "correct", "incorrect", "ambiguous"), confidence (0-1).
"""


@lm(scripted_crag_model, system_prompt=EVALUATE_PROMPT)
def evaluate_retrieval(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Judge relevance of each retrieved document."""
    retrieved = ctx.trace[RETRIEVE].value
    payload = json.dumps(
        {
            "question": retrieved["question"],
            "documents": retrieved["documents"],
        }
    )
    raw = call(messages=[{"role": "user", "content": payload}])
    evaluations = json.loads(strip_fences(raw))
    n_correct = sum(1 for e in evaluations if e["label"] == "correct")
    n_incorrect = sum(1 for e in evaluations if e["label"] == "incorrect")
    avg_confidence = sum(e["confidence"] for e in evaluations) / (len(evaluations) or 1)
    return StepResult(
        value={"evaluations": evaluations, "avg_confidence": avg_confidence},
        meta={
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_ambiguous": len(evaluations) - n_correct - n_incorrect,
        },
    )


def select_action(ctx: SkillContext) -> StepResult:
    """Route based on retrieval quality: correct / incorrect / ambiguous."""
    eval_result = ctx.trace[EVALUATE].value
    avg_conf = eval_result["avg_confidence"]
    evaluations = eval_result["evaluations"]
    if avg_conf >= CONFIDENCE_UPPER:
        action = "correct"
    elif avg_conf <= CONFIDENCE_LOWER:
        action = "incorrect"
    else:
        action = "ambiguous"
    kept_titles = [
        e["title"] for e in evaluations if e["label"] in ("correct", "ambiguous")
    ]
    return StepResult(
        value={
            "action": action,
            "kept_titles": kept_titles,
            "avg_confidence": avg_conf,
        },
        meta={"action": action},
    )


def refine_documents(ctx: SkillContext) -> StepResult:
    """Decompose-then-recompose: keep only relevant sentences from kept docs."""
    action_result = ctx.trace[ACTION].value
    action = action_result["action"]
    question = ctx.trace[PARSE].value["question"]
    if action == "incorrect":
        return StepResult(
            value={"context": "", "action": action},
            meta={"kept_sentences": 0, "total_sentences": 0},
        )
    retrieved_docs = ctx.trace[RETRIEVE].value["documents"]
    kept_titles = set(action_result["kept_titles"])
    q_tokens = set(_tokenize(question))
    kept_sentences: list[str] = []
    total_sentences = 0
    for doc in retrieved_docs:
        if doc["title"] not in kept_titles:
            continue
        for sent in doc["sentences"]:
            total_sentences += 1
            s_tokens = set(_tokenize(sent))
            overlap = len(q_tokens & s_tokens) / (len(q_tokens) or 1)
            if overlap >= 0.15:
                kept_sentences.append(sent.strip())
    context = " ".join(kept_sentences) if kept_sentences else ""
    return StepResult(
        value={"context": context, "action": action},
        meta={
            "kept_sentences": len(kept_sentences),
            "total_sentences": total_sentences,
        },
    )


GENERATE_PROMPT = """\
You are a QA system. Answer the question using ONLY the provided context.
If the context is empty, answer from your own knowledge but indicate low confidence.
Return ONLY JSON: {"answer": "<short factoid answer>", "confidence": <0-1>}
"""


@lm(scripted_crag_model, system_prompt=GENERATE_PROMPT)
def generate_answer(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Generate final short-form answer from refined context."""
    refined = ctx.trace[REFINE].value
    question = ctx.trace[PARSE].value["question"]
    payload = json.dumps({"question": question, "context": refined["context"]})
    raw = call(messages=[{"role": "user", "content": payload}])
    parsed = json.loads(strip_fences(raw))
    answer = parsed.get("answer", "").strip()
    confidence = parsed.get("confidence", 0.0)
    return StepResult(
        value={"answer": answer, "action": refined["action"]},
        meta={
            "confidence": confidence,
            "parametric_only": refined["action"] == "incorrect",
        },
    )


crag_solver = Skill(
    name="crag_solver",
    steps=[
        Skill(PARSE, fn=parse_query),
        Skill(RETRIEVE, fn=retrieve),
        Skill(EVALUATE, fn=evaluate_retrieval),
        Skill(ACTION, fn=select_action),
        Skill(REFINE, fn=refine_documents),
        Skill(GENERATE, fn=generate_answer),
    ],
)

__all__ = [
    "ACTION",
    "EVALUATE",
    "GENERATE",
    "MODEL",
    "PARSE",
    "REFINE",
    "RETRIEVE",
    "crag_solver",
    "run_skill",
    "scripted_crag_model",
]
