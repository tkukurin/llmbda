# %%
"""Inspect scoring for the support triage skill."""

import os
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    Metric,
    Score,
    Target,
    accuracy,
    match,
    mean,
    metric,
    model_graded_qa,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from tk.llmbda.inspect import skill_solver, step_scorer

from .skill import (
    CLASSIFY,
    DRAFT,
    IDENTIFIERS,
    SUMMARIZE,
    TICKETS,
    support_triage,
)

_LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs")

# %%
EXPECTED = {
    "SUP-1001": ("billing_refund", "P2"),
    "SUP-1002": ("production_incident", "P0"),
    "SUP-1003": ("account_access", "P1"),
}

EVAL_SAMPLES = [
    Sample(
        id=t["id"],
        input=t["subject"],
        target=list(EXPECTED[t["id"]]),
        metadata={"ticket": t},
    )
    for t in TICKETS
]


# %%
def _trace(state: TaskState) -> dict:
    return (state.metadata or {}).get("llmbda.trace", {})


classify_matches_intent = step_scorer(
    CLASSIFY,
    match(location="exact"),
    project=lambda v: v["intent"],
)


@scorer(metrics=[accuracy(), stderr()])
def draft_priority_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        got = _trace(state)[DRAFT].value["priority"]
        want = target[1]
        return Score(
            value="C" if got == want else "I",
            answer=got,
            explanation=f"expected {want!r}, got {got!r}",
        )

    return score


# %%
REPLY_QUALITY_TEMPLATE = """\
You are evaluating a customer support reply for quality.

[BEGIN DATA]
***
[Customer request]: {question}
***
[Support reply]: {answer}
***
[END DATA]

Grade the reply as CORRECT if it:
- Acknowledges the customer's specific issue
- Is professional and actionable
- Requests missing information when identifiers are absent

Grade as INCORRECT if the reply is generic, dismissive, or ignores
key details from the request.

{instructions}
"""


# %%
_ISSUE_KW = ["refund", "charge", "outage", "escalat", "access", "restore"]


@scorer(metrics=[mean(), stderr()])
def draft_reply_heuristic():
    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        tr = _trace(state)
        reply = tr[DRAFT].value.get("customer_reply", "").lower()
        missing_ids = not tr[IDENTIFIERS].value["account_ids"]
        ack = 0.5 if any(kw in reply for kw in _ISSUE_KW) else 0.0
        info = (0.5 if "account" in reply else 0.0) if missing_ids else 0.5
        pts = ack + info
        reasons = []
        if ack:
            reasons.append("acknowledges issue")
        else:
            reasons.append("generic reply")
        if missing_ids:
            msg = "requests missing id" if info else "missing id not requested"
            reasons.append(msg)
        else:
            reasons.append("no missing ids")
        return Score(value=pts, answer=reply, explanation="; ".join(reasons))

    return score


# %%
@metric
def strict_accuracy() -> Metric:
    def m(scores: list) -> float:
        if not scores:
            return 0.0
        return sum(float(s.score.value) == 1.0 for s in scores) / len(scores)

    return m


@scorer(metrics=[accuracy(), stderr(), strict_accuracy()])
def final_status_scorer():
    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        status = _trace(state)[SUMMARIZE].value.get("status")
        return Score(
            value="C" if status == "validated" else "I",
            answer=str(status),
            explanation=f"status={status!r}",
        )

    return score


# %%
def build_task(model: str, limit: int | None = None) -> Task:  # noqa: ARG001
    """Build Inspect evaluation task for support triage."""
    samples = EVAL_SAMPLES[:limit] if limit else EVAL_SAMPLES
    scorers = [
        classify_matches_intent,
        draft_priority_scorer(),
        draft_reply_heuristic(),
        final_status_scorer(),
    ]
    if grader := os.environ.get("INSPECT_GRADER"):
        g = model_graded_qa(template=REPLY_QUALITY_TEMPLATE, model=grader)
        quality = step_scorer(DRAFT, g, project=lambda v: v["customer_reply"])
        scorers.insert(2, quality)
    return Task(
        name="support_triage_eval",
        dataset=samples,
        solver=skill_solver(support_triage, entry=lambda s: s.metadata["ticket"]),
        scorer=scorers,
    )


# %%
if __name__ == "__main__":
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import EvalLog

    inspect_model = os.environ.get("INSPECT_MODEL", "none/none")
    task = build_task(model=inspect_model)

    eval_logs = inspect_eval(
        task,
        model=inspect_model,
        display="none",
        log_dir=_LOG_DIR,
    )
    assert isinstance((log := eval_logs[0]), EvalLog), f"{log=}"

    print(f"status: {log.status}")
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

    assert log.samples is not None
    for sample in log.samples:
        print(f"\n{sample.id}")
        assert sample.scores is not None
        for name, sc in sample.scores.items():
            print(f"  {name:28s} {sc.value}  ({sc.explanation})")
