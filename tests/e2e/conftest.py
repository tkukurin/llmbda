import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture
def openai_caller() -> Callable[..., str]:

    def _call(**kwargs: Any) -> str:
        api_key = os.environ["OPENAI_API_KEY"]
        payload = json.dumps(
            {"model": "gpt-4o-mini", "temperature": 0.0, **kwargs}
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req) as response:  # noqa: S310 (https only)
            res = json.loads(response.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"]

    return _call
