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
"""Inspect AI scoring for the GSM8K solver skill.

Run:  GSM8K_MODEL=openai/gpt-4o-mini uv run python examples/gsm8k/scoring.py
Limit: GSM8K_LIMIT=50 uv run python examples/gsm8k/scoring.py
View:  uv run inspect view
"""

import os

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import EvalLog
from inspect_ai.scorer import match
from skill import EXTRACT, MODEL, REPAIR, VERIFY, gsm8k_solver

from tk.llmbda.inspect import skill_solver, step_check, step_scorer

# %%
_ANSWER_DELIM = "####"


def _record_to_sample(record: dict) -> Sample:
    parts = record["answer"].split(_ANSWER_DELIM)
    target = parts[-1].strip().replace(",", "")
    return Sample(
        input=record["question"],
        target=target,
        metadata={"reasoning": _ANSWER_DELIM.join(parts[:-1]).strip()},
    )


EVAL_SAMPLES = hf_dataset(
    path="openai/gsm8k",
    data_dir="main",
    split="test",
    sample_fields=_record_to_sample,
    limit=int(os.environ.get("GSM8K_LIMIT", "0")) or None,
)


# %%
extraction_match = step_scorer(
    EXTRACT, match(numeric=True), project=lambda v: v["answer"]
)

final_match = step_scorer(REPAIR, match(numeric=True), project=lambda v: v["answer"])

arithmetic_validity = step_check(VERIFY, lambda r: r.meta.get("valid", False))


# %%
eval_task = Task(
    name="gsm8k_skill_eval",
    dataset=EVAL_SAMPLES,
    solver=skill_solver(gsm8k_solver, entry=lambda s: s.input_text),
    scorer=[extraction_match, arithmetic_validity, final_match],
)

print(f"model: {MODEL}, samples: {len(EVAL_SAMPLES)}")
eval_logs = inspect_eval(eval_task, model="none/none")
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
