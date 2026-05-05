# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "simple-parsing>=0.1",
#     "litellm>=1.0",
#     "tk-llmbda",
# ]
#
# [tool.uv.sources]
# tk-llmbda = { path = "../", editable = true }
# ///
"""Central experiment runner — model config lives here, not in each example."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import crag
import gsm8k
import simple_parsing as sp
import triage

Experiment = Literal["crag", "gsm8k", "triage"]
EXPERIMENTS = {"crag": crag.runxp, "gsm8k": gsm8k.runxp, "triage": triage.runxp}


@dataclass
class Args:
    """Run an llmbda example experiment."""
    experiment: Experiment = sp.field(positional=True)
    model: str = os.environ.get("LLMBDA_MODEL", "openai/gpt-4o-mini")


def getargs(**overrides) -> Args:
    """Parse from CLI in scripts, use overrides interactively."""
    interactive = not hasattr(sys.modules.get("__main__"), "__file__") or "ipykernel" in sys.modules
    return Args(**overrides) if interactive else sp.parse(Args)


if __name__ == "__main__":
    args = getargs()
    EXPERIMENTS[args.experiment](args.model)
