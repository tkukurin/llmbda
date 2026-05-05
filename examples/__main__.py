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
  uv run examples/__main__.py crag
  uv run examples/__main__.py gsm8k --model openai/gpt-4o
  uv run examples/__main__.py triage --score
  LLMBDA_MODEL=openai/gpt-4o uv run examples/__main__.py crag --score --limit 50
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import simple_parsing as sp

Experiment = Literal["crag", "gsm8k", "triage"]


@dataclass
class Args:
    """Run llmbda experiments (quick demo or full Inspect AI scoring).

    Experiments:
      crag    — retrieval relevance + QA generation (2-step CRAG)
      gsm8k   — chain-of-thought math with self-verification
      triage  — support ticket classification, routing, repair loop
    """
    experiment: Experiment = sp.field(positional=True)  # which experiment to run
    model: str = os.environ.get("LLMBDA_MODEL", "openai/gpt-4o-mini")  # LLM model
    score: bool = False  # run Inspect AI evaluation instead of quick demo
    limit: int = 0  # max eval samples (0 = unlimited, only with --score)


def getargs(**overrides) -> Args:
    """Parse from CLI in scripts, use overrides interactively."""
    interactive = (
        not hasattr(sys.modules.get("__main__"), "__file__")
        or "ipykernel" in sys.modules
    )
    return Args(**overrides) if interactive else sp.parse(Args)


def _run_demo(args: Args):
    import crag
    import gsm8k
    import triage
    {"crag": crag.runxp, "gsm8k": gsm8k.runxp, "triage": triage.runxp}[
        args.experiment
    ](args.model)


def _run_scoring(args: Args):
    from pathlib import Path

    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import EvalLog

    log_dir = str(Path(__file__).resolve().parents[1] / "logs")
    limit = args.limit or None

    if args.experiment == "crag":
        from crag.scoring import build_task
        task = build_task(args.model, limit=limit)
    elif args.experiment == "gsm8k":
        from gsm8k.scoring import build_task
        task = build_task(args.model, limit=limit)
    elif args.experiment == "triage":
        from triage.scoring import build_task
        task = build_task(args.model, limit=limit)

    print(f"model={args.model} experiment={args.experiment} limit={limit}")
    inspect_model = os.environ.get("INSPECT_MODEL", args.model)
    logs = inspect_eval(task, model=inspect_model, log_dir=log_dir)
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
    if args.score:
        _run_scoring(args)
    else:
        _run_demo(args)
