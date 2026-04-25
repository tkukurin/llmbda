from tk.llmbda import strip_fences


def test_json_fence():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

def test_bare_fence():
    assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'

def test_one_line_fence():
    assert strip_fences('```{"a": 1}```') == '{"a": 1}'

def test_no_fence():
    assert strip_fences('{"a": 1}') == '{"a": 1}'

def test_strips_whitespace():
    assert strip_fences('  \n```json\n{"a": 1}\n```\n  ') == '{"a": 1}'

def test_mixed_case_json_fence():
    assert strip_fences('```Json\n{"a": 1}\n```') == '{"a": 1}'

def test_spaced_json_fence():
    assert strip_fences('``` json\n{"a": 1}\n```') == '{"a": 1}'

def test_jsonc_fence():
    assert strip_fences('```jsonc\n{"a": 1}\n```') == '{"a": 1}'

def test_non_json_fence():
    assert strip_fences('```python\n{"a": 1}\n```') == '{"a": 1}'

def test_partial_fence_is_left_unchanged():
    text = 'prefix ```json\n{"a": 1}\n``` suffix'
    assert strip_fences(text) == text
