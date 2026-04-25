"""Utilities for wrangling LLM output.

Functions here are deliberately stateless and side-effect-free --
pure transforms on strings.  (Fitting for a library named after
lambda calculus.)
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*\n?(.*?)\n?\s*```$",
    re.DOTALL,
)


def strip_fences(text: str) -> str:
    r"""Remove Markdown code fences that LLMs love to add.

    >>> strip_fences('```json\n{"a": 1}\n```')
    '{"a": 1}'
    >>> strip_fences('{"a": 1}')
    '{"a": 1}'
    """
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Strip fences and parse JSON from an LLM response.

    Raises :class:`json.JSONDecodeError` if the content is not valid
    JSON after fence removal.
    """
    return json.loads(strip_fences(raw))
