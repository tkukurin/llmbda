from __future__ import annotations

import json
import os
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def openai_caller() -> Callable[..., str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for OpenAI e2e tests")

    def _call(**kwargs: Any) -> str:
        payload = json.dumps(
            {"model": "gpt-4o-mini", "temperature": 0.0, **kwargs},
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
            res = json.loads(response.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]

    return _call
