"""Utilities for wrangling LLM output.

Functions here are deliberately stateless and side-effect-free --
pure transforms on strings.  (Fitting for a library named after
lambda calculus.)
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*\n?(.*?)\n?\s*```$",
    re.DOTALL,
)


def strip_fences(text: str) -> str:
    """Remove Markdown code fences.

    >>> strip_fences('```json\n{"a": 1}\n```')
    '{"a": 1}'
    >>> strip_fences('{"a": 1}')
    '{"a": 1}'
    """
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text
