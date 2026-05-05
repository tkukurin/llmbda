# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "simple-parsing>=0.1",
#     "litellm>=1.0",
#     "inspect-ai>=0.3",
#     "datasets",
#     "tk-llmbda[inspect]",
# ]
#
# [tool.uv.sources]
# tk-llmbda = { path = "../", editable = true }
# ///
"""Run llmbda example experiments.

Examples:
  uv run examples/cli.py crag
  uv run examples/cli.py gsm8k --model openai/gpt-4o
  uv run examples/cli.py triage --limit 5
  LLMBDA_MODEL=openai/gpt-4o uv run examples/cli.py crag --limit 50
"""
from __future__ import annotations

import dataclasses as dc
import os
import sys
from pathlib import Path
from typing import Literal

import importlib

import simple_parsing as sp
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

Experiment = Literal["crag", "gsm8k", "triage"]


@dc.dataclass
class Args:
    """Run llmbda experiments via Inspect AI scoring.

    Experiments:
      crag    — retrieval relevance + QA generation (2-step CRAG)
      gsm8k   — chain-of-thought math with self-verification
      triage  — support ticket classification, routing, repair loop
    """
    experiment: Experiment = sp.field(positional=True)
    model: str = os.environ.get("LLMBDA_MODEL", "openai/gpt-4o-mini")
    limit: int = 1


def getargs(**overrides) -> Args:
    """Parse from CLI in scripts, use overrides interactively."""
    interactive = (
        not hasattr(sys.modules.get("__main__"), "__file__")
        or "ipykernel" in sys.modules
    )
    return Args(**overrides) if interactive else sp.parse(Args)


def _run_scoring(args: Args):
    log_dir = str(Path(__file__).resolve().parents[1] / "logs")
    module = importlib.import_module(f"{args.experiment}.scoring")
    builder = module.build_task
    task = builder(args.model, limit=args.limit)
    print(dc.asdict(args))
    logs = inspect_eval(task, model=args.model, log_dir=log_dir)
    assert isinstance((log := logs[0]), EvalLog)
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


if __name__ == "__main__":
    args = getargs()
    _run_scoring(args)
