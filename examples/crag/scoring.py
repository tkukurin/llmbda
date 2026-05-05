# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm>=1.0",
#     "inspect-ai>=0.3",
#     "datasets",
#     "tk-llmbda[inspect]",
# ]
#
# [tool.uv.sources]
# tk-llmbda = { path = "../../", editable = true }
# ///
# %%
"""Inspect scoring for the CRAG skill."""

import math
import os
import re
from collections import Counter
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import EvalLog
from inspect_ai.scorer import Score, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState

from tk.llmbda.inspect import skill_solver

from .skill import (
    GENERATE,
    make_skill,
)

_LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs")

# %%
_TOP_K = int(os.environ.get("CRAG_TOP_K", "5"))
_WORD_RE = re.compile(r"\w+")


def _build_paragraphs(record: dict) -> list[dict]:
    """Extract title+text+sentences from a HotpotQA record."""
    titles = record["context"]["title"]
    sentences_list = record["context"]["sentences"]
    return [
        {
            "title": t,
            "text": " ".join(s.strip() for s in sents),
            "sentences": list(sents),
        }
        for t, sents in zip(titles, sentences_list, strict=False)
    ]


def _retrieve_topk(
    question: str, paragraphs: list[dict], k: int = _TOP_K
) -> list[dict]:
    """TF-IDF ranking of paragraphs, returns top-k as documents."""
    q_tokens = [w.lower() for w in _WORD_RE.findall(question)]
    corpus_tokens = [
        [w.lower() for w in _WORD_RE.findall(p["text"])] for p in paragraphs
    ]
    n = len(corpus_tokens) or 1
    df: Counter[str] = Counter()
    for doc_toks in corpus_tokens:
        df.update(set(doc_toks))
    idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
    scores = []
    for i, doc_toks in enumerate(corpus_tokens):
        tf = Counter(doc_toks)
        total = len(doc_toks) or 1
        score = sum((tf[t] / total) * idf.get(t, 0.0) for t in set(q_tokens))
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [paragraphs[i] for i, _ in scores[:k]]


def _record_to_sample(record: dict) -> Sample:
    """Build Sample with pre-retrieved documents in metadata."""
    question = record["question"].strip()
    paragraphs = _build_paragraphs(record)
    documents = _retrieve_topk(question, paragraphs)
    return Sample(
        id=record["id"],
        input=question,
        target=record["answer"],
        metadata={
            "question": question,
            "documents": documents,
            "gold_titles": list(set(record["supporting_facts"]["title"])),
        },
    )


# %%


def _normalize(text: str) -> list[str]:
    """Lowercase, strip articles/punctuation, tokenize."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction)
    ref_tokens = _normalize(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@scorer(metrics=[mean(), stderr()])
def answer_f1():
    """Token-level F1 between generated answer and gold answer."""

    async def score(state: TaskState, target: Target) -> Score:
        trace = (state.metadata or {}).get("llmbda.trace", {})
        answer = trace[GENERATE].value["answer"] if GENERATE in trace else ""
        f1 = _token_f1(answer, target.text)
        return Score(value=f1, answer=answer, explanation=f"F1={f1:.3f}")

    return score


@scorer(metrics=[accuracy(), stderr()])
def answer_em():
    """Exact match between generated answer and gold (normalized)."""

    async def score(state: TaskState, target: Target) -> Score:
        trace = (state.metadata or {}).get("llmbda.trace", {})
        answer = trace[GENERATE].value["answer"] if GENERATE in trace else ""
        pred_norm = " ".join(_normalize(answer))
        gold_norm = " ".join(_normalize(target.text))
        match = pred_norm == gold_norm
        return Score(value="C" if match else "I", answer=answer)

    return score


# %%
def build_task(model: str, limit: int | None = None) -> Task:
    """Create the CRAG evaluation task for a given model and optional sample limit."""
    samples = hf_dataset(
        path="hotpot_qa",
        name="distractor",
        split="validation",
        sample_fields=_record_to_sample,
        limit=limit,
    )
    return Task(
        name="crag_hotpotqa",
        dataset=samples,
        solver=skill_solver(
            make_skill(model),
            entry=lambda s: {
                "question": s.metadata["question"],
                "documents": s.metadata["documents"],
            },
        ),
        scorer=[answer_f1(), answer_em()],
    )


if __name__ == "__main__":
    model = os.environ.get("LLMBDA_MODEL", "openai/gpt-4o-mini")
    inspect_model = os.environ.get("INSPECT_MODEL", model)
    limit = int(os.environ.get("CRAG_LIMIT", "0")) or None

    eval_task = build_task(model, limit=limit)
    print(f"model: {model}, inspect_model: {inspect_model}, limit: {limit}")
    eval_logs = inspect_eval(eval_task, model=inspect_model, log_dir=_LOG_DIR)
    assert isinstance((log := eval_logs[0]), EvalLog), f"{log=}"

    print(f"\nstatus: {log.status}")
    if log.status != "success":
        if log.error:
            print(f"error: {log.error.message}")
            if log.error.traceback:
                print(log.error.traceback)
        raise SystemExit(1)

    assert log.results is not None
    for sr in log.results.scores:
        print(f"\n{sr.name}")
        for name, mr in sr.metrics.items():
            print(f"  {name:16s} = {mr.value:.3f}")
