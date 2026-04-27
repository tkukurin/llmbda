from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tk.llmbda import aiter_skill, arun_skill, iter_skill, run_skill

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@pytest.fixture(params=["sync", "async"])
def run(request):
    """run(skill, entry) -> SkillResult, parametrized over sync and async paths."""
    if request.param == "sync":
        return run_skill
    return lambda skill, entry: asyncio.run(arun_skill(skill, entry))


@pytest.fixture(params=["sync", "async"])
def iter_collect(request):
    """iter_collect(skill, entry) -> list[(name, StepResult)], parametrized."""
    if request.param == "sync":
        return lambda skill, entry: list(iter_skill(skill, entry))

    async def _collect(skill, entry):
        return [item async for item in aiter_skill(skill, entry)]

    return lambda skill, entry: asyncio.run(_collect(skill, entry))


@pytest.fixture
def scripted_caller() -> Callable[[Iterable[Any]], Callable[..., Any]]:
    """Build a fake caller that yields *responses* in order on successive calls."""

    def _make(responses: Iterable[Any]) -> Callable[..., Any]:
        it = iter(responses)

        def _call(**_kw: Any) -> Any:
            return next(it)

        return _call

    return _make
