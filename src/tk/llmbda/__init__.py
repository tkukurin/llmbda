"""tk.llmbda -- skill composition for LLMs.

Like lambda calculus, but the functions talk back.
"""

from .compose import compose, run_skill
from .skill import Caller, Skill, Step, StepContext, StepResult
from .utils import parse_llm_json, strip_fences

__all__ = [
    "Caller",
    "Skill",
    "Step",
    "StepContext",
    "StepResult",
    "compose",
    "parse_llm_json",
    "run_skill",
    "strip_fences",
]
