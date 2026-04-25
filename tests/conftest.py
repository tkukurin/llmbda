from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@pytest.fixture
def scripted_caller() -> Callable[[Iterable[Any]], Callable[..., Any]]:
    """Build a fake caller that yields *responses* in order on successive calls."""
    def _make(responses: Iterable[Any]) -> Callable[..., Any]:
        it = iter(responses)
        def _call(**_kw: Any) -> Any:
            return next(it)
        return _call
    return _make
