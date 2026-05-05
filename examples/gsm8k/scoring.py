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
"""Inspect evaluation for the GSM8K solver skill.

- Run: `GSM8K_MODEL=openai/gpt-4o-mini uv run examples/gsm8k/scoring.py`
- Limit: `GSM8K_LIMIT=50 GSM8K_MODEL=openai/gpt-4o-mini uv run examples/gsm8k/scoring.py`
- View: `uv run inspect view`
"""

from __future__ import annotations

import os
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import EvalLog
from inspect_ai.scorer import match
from skill import MODEL, gsm8k

from tk.llmbda.inspect import skill_solver

_LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs")
INSPECT_MODEL = os.environ.get("INSPECT_MODEL", MODEL)

# %%
_ANSWER_DELIM = "####"


def _record_to_sample(record: dict) -> Sample:
    """Extract question and numeric target from GSM8K record."""
    target = record["answer"].split(_ANSWER_DELIM)[-1].strip().replace(",", "")
    return Sample(input=record["question"], target=target)


EVAL_SAMPLES = hf_dataset(
    path="openai/gsm8k",
    data_dir="main",
    split="test",
    sample_fields=_record_to_sample,
    limit=int(os.environ.get("GSM8K_LIMIT", "0")) or None,
)

# %%
eval_task = Task(
    name="gsm8k",
    dataset=EVAL_SAMPLES,
    solver=skill_solver(gsm8k, entry=lambda s: s.input_text),
    scorer=match(numeric=True),
)

# %%
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
