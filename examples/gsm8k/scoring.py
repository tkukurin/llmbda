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
"""Inspect evaluation for the GSM8K solver skill."""
from __future__ import annotations

import os
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import EvalLog
from inspect_ai.scorer import match

from tk.llmbda.inspect import skill_solver

from .skill import make_skill

_ANSWER_DELIM = "####"


def _record_to_sample(record: dict) -> Sample:
    """Extract question and numeric target from GSM8K record."""
    target = record["answer"].split(_ANSWER_DELIM)[-1].strip().replace(",", "")
    return Sample(input=record["question"], target=target)


def build_task(model: str, limit: int | None = None) -> Task:
    """Build Inspect evaluation task for GSM8K."""
    samples = hf_dataset(
        path="openai/gsm8k",
        data_dir="main",
        split="test",
        sample_fields=_record_to_sample,
        limit=limit,
    )
    return Task(
        name="gsm8k",
        dataset=samples,
        solver=skill_solver(make_skill(model), entry=lambda s: s.input_text),
        scorer=match(numeric=True),
    )


# %%
if __name__ == "__main__":
    _LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs")
    _MODEL = os.environ.get("LLMBDA_MODEL", "openai/gpt-4o-mini")
    _INSPECT_MODEL = os.environ.get("INSPECT_MODEL", _MODEL)
    _LIMIT = int(os.environ.get("GSM8K_LIMIT", "0")) or None

    eval_task = build_task(_MODEL, limit=_LIMIT)
    n = len(eval_task.dataset)
    print(f"model: {_MODEL}, inspect_model: {_INSPECT_MODEL}, samples: {n}")
    eval_logs = inspect_eval(eval_task, model=_INSPECT_MODEL, log_dir=_LOG_DIR)
    assert isinstance((log := eval_logs[0]), EvalLog)

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
