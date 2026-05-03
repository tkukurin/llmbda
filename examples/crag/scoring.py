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
"""Inspect scoring for the CRAG solver skill.

- Run: `uv run examples/crag/scoring.py`
- LLM: `CRAG_MODEL=openai/gpt-4o-mini CRAG_LIMIT=50 uv run examples/crag/scoring.py`
- View: `uv run inspect view`
"""

import os
from collections import Counter
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import EvalLog
from inspect_ai.scorer import (
    Score,
    Target,
    accuracy,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from skill import (
    ACTION,
    EVALUATE,
    GENERATE,
    MODEL,
    RETRIEVE,
    crag_solver,
    scripted_crag_model,
)

from tk.llmbda.inspect import passthrough_model, skill_solver

_LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs")
_SCRIPTED = passthrough_model(scripted_crag_model, name="crag")
INSPECT_MODEL = os.environ.get("INSPECT_MODEL", _SCRIPTED)

# %%
_LIMIT = int(os.environ.get("CRAG_LIMIT", "0")) or None


def _record_to_sample(record: dict) -> Sample:
    """Convert a HotpotQA record into an Inspect Sample."""
    gold_titles = set(record["supporting_facts"]["title"])
    return Sample(
        id=record["id"],
        input=record["question"],
        target=record["answer"],
        metadata={
            "record": record,
            "gold_titles": list(gold_titles),
            "type": record["type"],
            "level": record["level"],
        },
    )


EVAL_SAMPLES = hf_dataset(
    path="hotpot_qa",
    name="distractor",
    split="validation",
    sample_fields=_record_to_sample,
    limit=_LIMIT,
)


# %%


def _normalize(text: str) -> list[str]:
    """Lowercase, strip articles/punctuation, tokenize."""
    import re  # noqa: PLC0415

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
        gold = target.text
        f1 = _token_f1(answer, gold)
        return Score(
            value=f1,
            answer=answer,
            explanation=f"F1={f1:.3f} (pred={answer!r}, gold={gold!r})",
        )

    return score


# %%


@scorer(metrics=[accuracy(), stderr()])
def answer_em():
    """Exact match between generated answer and gold (normalized)."""

    async def score(state: TaskState, target: Target) -> Score:
        trace = (state.metadata or {}).get("llmbda.trace", {})
        answer = trace[GENERATE].value["answer"] if GENERATE in trace else ""
        pred_norm = " ".join(_normalize(answer))
        gold_norm = " ".join(_normalize(target.text))
        match = pred_norm == gold_norm
        return Score(
            value="C" if match else "I",
            answer=answer,
            explanation=f"pred={pred_norm!r}, gold={gold_norm!r}",
        )

    return score


# %%


@scorer(metrics=[mean(), stderr()])
def retrieval_eval_quality():
    """Did the evaluator correctly identify gold supporting paragraphs?"""

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        trace = (state.metadata or {}).get("llmbda.trace", {})
        gold_titles = set(state.metadata.get("gold_titles", []))
        if EVALUATE not in trace or not gold_titles:
            return Score(value=0.0, explanation="missing trace or gold_titles")
        evaluations = trace[EVALUATE].value["evaluations"]
        kept = {
            e["title"] for e in evaluations if e["label"] in ("correct", "ambiguous")
        }
        retrieved_titles = {e["title"] for e in evaluations}
        gold_in_retrieved = gold_titles & retrieved_titles
        if not gold_in_retrieved:
            return Score(value=0.0, explanation="no gold docs in top-k")
        recalled = gold_in_retrieved & kept
        recall = len(recalled) / len(gold_in_retrieved)
        return Score(
            value=recall,
            answer=str(sorted(kept)),
            explanation=(
                f"recall={recall:.2f}"
                f" ({len(recalled)}/{len(gold_in_retrieved)} gold docs kept)"
            ),
        )

    return score


# %%


@scorer(metrics=[mean(), stderr()])
def action_correctness():
    """Was the action reasonable given gold doc presence in retrieved set?"""

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        trace = (state.metadata or {}).get("llmbda.trace", {})
        gold_titles = set(state.metadata.get("gold_titles", []))
        if ACTION not in trace or RETRIEVE not in trace:
            return Score(value=0.0, explanation="missing trace")
        action = trace[ACTION].value["action"]
        retrieved_titles = {d["title"] for d in trace[RETRIEVE].value["documents"]}
        gold_in_retrieved = gold_titles & retrieved_titles
        has_gold = len(gold_in_retrieved) > 0
        if has_gold and action in ("correct", "ambiguous"):
            val = 1.0
            reason = f"gold present, action={action} (good)"
        elif not has_gold and action == "incorrect":
            val = 1.0
            reason = "no gold in top-k, action=incorrect (good)"
        elif has_gold and action == "incorrect":
            val = 0.0
            reason = "gold present but discarded (bad)"
        else:
            val = 0.5
            reason = f"no gold in top-k, action={action} (acceptable)"
        return Score(value=val, answer=action, explanation=reason)

    return score


# %%


eval_task = Task(
    name="crag_hotpotqa_eval",
    dataset=EVAL_SAMPLES,
    solver=skill_solver(crag_solver, entry=lambda s: s.metadata["record"]),
    scorer=[
        retrieval_eval_quality(),
        action_correctness(),
        answer_f1(),
        answer_em(),
    ],
)

print(f"model: {MODEL}, inspect_model: {INSPECT_MODEL}, samples: {len(EVAL_SAMPLES)}")
eval_logs = inspect_eval(eval_task, model=INSPECT_MODEL, log_dir=_LOG_DIR)
assert isinstance((log := eval_logs[0]), EvalLog), f"{log=}"  # noqa: RUF018

# %%
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
