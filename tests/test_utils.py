"""Tests for LLM output utilities."""

from tk.llmbda import parse_llm_json, strip_fences


class TestStripFences:
    def test_json_fence(self):
        assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_bare_fence(self):
        assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_no_fence(self):
        assert strip_fences('{"a": 1}') == '{"a": 1}'

    def test_strips_whitespace(self):
        assert strip_fences('  \n```json\n{"a": 1}\n```\n  ') == '{"a": 1}'


class TestParseLlmJson:
    def test_fenced_json(self):
        result = parse_llm_json('```json\n{"key": "val"}\n```')
        assert result == {"key": "val"}

    def test_bare_json(self):
        result = parse_llm_json('{"key": "val"}')
        assert result == {"key": "val"}
