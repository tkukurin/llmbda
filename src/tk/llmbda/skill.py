"""Core data structures for skill composition.

A Skill is a named sequence of Steps.  Each Step receives a
:class:`StepContext` and returns a :class:`StepResult`.  Steps execute
in order; a step can mark its result as *terminal* to short-circuit the
remaining steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

class Caller(Protocol):
    """Minimal contract for an LLM caller.
        lambda **kw: ...completions.create(**kw).choices[0].message.content
        lambda **kw: '{"answer": 42}'
    """
    def __call__(self, **kwargs: Any) -> str: ...


@dataclass
class StepResult:
    """What a single step produces.

    value: step output (arbitrary JSON-serialisable data).
    metadata: dict persisted alongside the value.
    terminal: If ``True``, :func:`run_skill` stops here.
    """

    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal: bool = False


@dataclass
class StepContext:
    """Accumulator threaded through the step sequence.

    entry: The raw data entry being processed.
    caller: LLM caller (``None`` for deterministic-only runs).
    system_prompt: The skill-level system prompt; LLM-using steps read this.
    prior: Results of previous steps, keyed by step name.
    """

    entry: dict[str, Any]
    caller: Caller | None = None
    system_prompt: str = ""
    prior: dict[str, StepResult] = field(default_factory=dict)


@dataclass
class Step:
    """A named operation: ``StepContext -> StepResult``."""

    name: str
    run: Callable[[StepContext], StepResult]


@dataclass
class Skill:
    """A named sequence of steps with a shared system prompt.

    Different skills define different workflow shapes -- the class
    imposes no fixed lifecycle.  A normalise skill might have two
    steps (deterministic parse, LLM fallback) while a refine skill
    has three (prepare, call LLM, validate + merge).
    """

    name: str
    description: str = ""
    system_prompt: str = ""
    steps: list[Step] = field(default_factory=list)
