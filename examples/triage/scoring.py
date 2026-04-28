# %% [markdown]
# # Inspect AI evaluation of the triage skill
#
# Wires `support_triage` (from `skill.py`) into an Inspect AI `Task` with
# per-step scorers. See the README's "Inspect AI integration" section for
# adapter docs.

# %%
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.scorer import (
    Metric,
    Score,
    Target,
    accuracy,
    match,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from skill import CLASSIFY, DRAFT, SUMMARIZE, TICKETS, support_triage

from tk.llmbda.inspect import skill_solver, step_scorer

# %% [markdown]
# ## Dataset
#
# `metadata["ticket"]` is the structured input. `target` is a list
# `[intent, priority]` — scorers that need a specific field index into it
# (`target[0]`, `target[1]`); `match()` against the list has any-of
# semantics.

# %%
EXPECTED = {
    "SUP-1001": ("billing_refund", "P2"),
    "SUP-1002": ("production_incident", "P0"),
    # SUP-1003 mis-expects P1 so one cell fails and aggregate metrics are non-trivial.
    "SUP-1003": ("account_access", "P1"),
}

EVAL_SAMPLES = [
    Sample(
        id=ticket["id"],
        input=ticket["subject"],
        target=list(EXPECTED[ticket["id"]]),
        metadata={"ticket": ticket},
    )
    for ticket in TICKETS
]

# %% [markdown]
# ## Scorers
#
# - `classify_matches_intent`: built-in `match()` wrapped by `step_scorer`
#   on the CLASSIFY step, projected to the `intent` field. Against the list
#   target, `match()` is any-of — the projected intent must equal one of the
#   target entries.
# - `draft_priority_scorer`: hand-written `@scorer` that indexes into
#   `target[1]` for the expected priority.
#
# TODO: add an LLM-graded scorer (e.g. `model_graded_qa()` wrapped by
# `step_scorer` on the DRAFT step's `customer_reply`) so the eval exercises
# a non-deterministic judge path, not just exact-match comparisons.

# %%
classify_matches_intent = step_scorer(
    CLASSIFY,
    match(location="exact"),
    project=lambda v: v["intent"],
)


@scorer(metrics=[accuracy(), stderr()])
def draft_priority_scorer():
    """Check draft step's priority against target[1]."""

    async def score(state: TaskState, target: Target) -> Score:
        trace = (state.metadata or {}).get("llmbda.trace", {})
        got = trace[DRAFT].value["priority"]
        want = target[1]
        return Score(
            value="C" if got == want else "I",
            answer=got,
            explanation=f"expected priority {want!r}, got {got!r}",
        )

    return score


# %% [markdown]
# ## Custom metric
#
# Inspect pre-casts score values to float before metric aggregation, so
# compare via `float(...)`, not `== "C"`.


# %%
@metric
def strict_accuracy() -> Metric:
    """Fraction of scores whose float value is exactly 1.0."""

    def m(scores: list) -> float:
        if not scores:
            return 0.0
        return sum(1.0 for s in scores if float(s.score.value) == 1.0) / len(scores)

    return m


@scorer(metrics=[accuracy(), stderr(), strict_accuracy()])
def final_status_scorer():
    """Check the skill resolved to `status == 'validated'` (target-independent)."""

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        trace = (state.metadata or {}).get("llmbda.trace", {})
        status = trace[SUMMARIZE].value.get("status")
        return Score(
            value="C" if status == "validated" else "I",
            answer=str(status),
            explanation=f"status={status!r}",
        )

    return score


# %% [markdown]
# ## Run
#
# `model="none/none"` is Inspect's no-op provider; `skill_solver` never calls
# `generate`. Logs land in `./logs/` — view with `uv run inspect view`.

# %%
eval_task = Task(
    name="support_triage_eval",
    dataset=EVAL_SAMPLES,
    solver=skill_solver(support_triage, entry=lambda s: s.metadata["ticket"]),
    scorer=[
        classify_matches_intent,
        draft_priority_scorer(),
        final_status_scorer(),
    ],
)

eval_logs = inspect_eval(eval_task, model="none/none", display="none")
assert isinstance((log := eval_logs[0]), EvalLog), f"{log=}"  # noqa: RUF018

# %% [markdown]
# ## Aggregate metrics

# %%
print(f"status: {log.status}")
if log.status != "success":
    if log.error:
        print(f"error: {log.error.message}")
        if log.error.traceback:
            print(log.error.traceback)
    raise SystemExit(1)

assert log.results is not None
for scorer_result in log.results.scores:
    print(f"\n{scorer_result.name}")
    for metric_name, metric_result in scorer_result.metrics.items():
        print(f"  {metric_name:16s} = {metric_result.value:.3f}")

# %% [markdown]
# ## Per-sample scores

# %%
assert log.samples is not None
for sample in log.samples:
    print(f"\n{sample.id}")
    assert sample.scores is not None
    for scorer_name, score in sample.scores.items():
        print(f"  {scorer_name:28s} {score.value}  ({score.explanation})")
