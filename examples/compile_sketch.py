# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tk-llmbda",
# ]
#
# [tool.uv.sources]
# tk-llmbda = { path = "../", editable = true }
# ///
# %% [markdown]
# # Compile sketch
#
# Walk a `Skill` tree and produce a flat, readable `skill.md`.
#
# - **Deterministic leaves** → source code with framework boilerplate
#   rewritten away (`ctx.entry["k"]` → `k` param, `StepResult` stripped).
# - **LLM leaves** → description + system prompt as prose.
# - **Composites / orchestrators** → transparent; children are flattened.
#   Orchestrator description and system prompt appear in the preamble.

# %%
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

from tk.llmbda import Skill


def _is_lm(skill: Skill) -> bool:
    return skill.fn is not None and hasattr(skill.fn, "lm_model")


def _source_for(fn: Any) -> str:
    """Raw dedented source of *fn* (unwraps @lm)."""
    target = getattr(fn, "__wrapped__", fn)
    try:
        return textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):
        return ""


def _sanitize(key: str) -> str:
    """Turn 'ψ::extract' into 'extract', 'λ::weekday' into 'weekday'."""
    name = key.rsplit("::", maxsplit=1)[-1]
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in name).strip("_")
    return name or "step"


class _CtxCollector(ast.NodeVisitor):
    """First pass: collect ctx.entry / ctx.trace / ctx.prev access patterns."""

    def __init__(self) -> None:
        self.entry_keys: list[str] = []
        self.trace_keys: list[str] = []
        self.prev_attrs: list[str] = []
        self._seen_entry: set[str] = set()
        self._seen_trace: set[str] = set()
        self._seen_prev: set[str] = set()

    def _is_ctx(self, node: ast.AST, attr: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ctx"
            and node.attr == attr
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr in ("value", "meta")
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ctx"
            and node.value.attr == "prev"
            and node.attr not in self._seen_prev
        ):
            self._seen_prev.add(node.attr)
            self.prev_attrs.append(node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value
            if self._is_ctx(node.value, "entry") and key not in self._seen_entry:
                self._seen_entry.add(key)
                self.entry_keys.append(key)
            elif self._is_ctx(node.value, "trace") and key not in self._seen_trace:
                self._seen_trace.add(key)
                self.trace_keys.append(key)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            inner = node.func.value
            key = node.args[0].value
            if self._is_ctx(inner, "entry") and key not in self._seen_entry:
                self._seen_entry.add(key)
                self.entry_keys.append(key)
            elif self._is_ctx(inner, "trace") and key not in self._seen_trace:
                self._seen_trace.add(key)
                self.trace_keys.append(key)
        self.generic_visit(node)


class _RewriteBody(ast.NodeTransformer):
    """Second pass: rewrite ctx accesses and StepResult returns."""

    def __init__(self) -> None:
        self._trace_aliases: dict[str, str] = {}
        self._prev_aliases: set[str] = set()

    @staticmethod
    def _trace_key(node: ast.AST) -> str | None:
        """Extract key from ctx.trace["k"] or ctx.trace.get("k")."""
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ctx"
            and node.value.attr == "trace"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            return node.slice.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "ctx"
            and node.func.value.attr == "trace"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
        return None

    @staticmethod
    def _is_prev(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ctx"
            and node.attr == "prev"
        )

    @staticmethod
    def _has_value(node: ast.AST | None) -> bool:
        return node is not None and not (
            isinstance(node, ast.Constant) and node.value is None
        )

    def _set_alias(self, name: str, value: ast.AST) -> None:
        if key := self._trace_key(value):
            self._trace_aliases[name] = key
            self._prev_aliases.discard(name)
            return
        if self._is_prev(value):
            self._prev_aliases.add(name)
            self._trace_aliases.pop(name, None)
            return
        self._trace_aliases.pop(name, None)
        self._prev_aliases.discard(name)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._set_alias(target.id, node.value)
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._set_alias(node.target.id, node.value)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.attr in ("value", "meta"):
            alias = node.value.id
            if alias in self._trace_aliases:
                name = _sanitize(self._trace_aliases[alias])
                if node.attr != "value":
                    name = f"{name}_{node.attr}"
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
            if alias in self._prev_aliases:
                return ast.copy_location(
                    ast.Name(id=f"prev_{node.attr}", ctx=ast.Load()),
                    node,
                )
        if node.attr in ("value", "meta"):
            key = self._trace_key(node.value)
            if key is not None:
                name = _sanitize(key)
                if node.attr != "value":
                    name = f"{name}_{node.attr}"
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ctx"
            and node.value.attr == "prev"
            and node.attr in ("value", "meta")
        ):
            return ast.copy_location(
                ast.Name(id=f"prev_{node.attr}", ctx=ast.Load()),
                node,
            )
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ctx"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            key = node.slice.value
            if node.value.attr == "entry":
                return ast.copy_location(ast.Name(id=key, ctx=ast.Load()), node)
            if node.value.attr == "trace":
                return ast.copy_location(
                    ast.Name(id=_sanitize(key), ctx=ast.Load()),
                    node,
                )
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "ctx"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            attr = node.func.value.attr
            key = node.args[0].value
            if attr == "entry":
                return ast.copy_location(ast.Name(id=key, ctx=ast.Load()), node)
            if attr == "trace":
                return ast.copy_location(
                    ast.Name(id=_sanitize(key), ctx=ast.Load()),
                    node,
                )
        self.generic_visit(node)
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        self.generic_visit(node)
        if (
            node.value
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "StepResult"
        ):
            args = node.value.args
            kw = {k.arg: k.value for k in node.value.keywords}
            value = args[0] if args else kw.get("value", ast.Constant(value=None))
            meta = args[1] if len(args) > 1 else kw.get("meta") or kw.get("metadata")
            if self._has_value(meta):
                assert meta is not None
                node.value = ast.Dict(
                    keys=[ast.Constant(value="value"), ast.Constant(value="meta")],
                    values=[value, meta],
                )
            else:
                node.value = value
        return node


def _build_params(collector: _CtxCollector) -> list[ast.arg]:
    return [
        *[ast.arg(arg=k) for k in collector.entry_keys],
        *[ast.arg(arg=_sanitize(k)) for k in collector.trace_keys],
        *[ast.arg(arg=f"prev_{a}") for a in collector.prev_attrs],
    ]


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:] or [ast.Pass()]
    return body


def rewrite_step_source(fn: Any) -> str | None:
    """Rewrite a step fn's source to remove framework boilerplate.

    - Replaces ctx.entry["key"] with a direct *key* parameter.
    - Replaces ctx.trace["name"].value with a *name* parameter.
    - Replaces ctx.prev.value with a *prev_value* parameter.
    - Unwraps StepResult(value, ...) returns to bare value returns.
    - Strips the docstring, decorators, and return annotation.
    """
    source = _source_for(fn)
    if not source:
        return "(source unavailable)"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source.strip()

    func_def: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        return source.strip()

    collector = _CtxCollector()
    collector.visit(func_def)

    func_def.args = ast.arguments(
        posonlyargs=[],
        args=_build_params(collector),
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    func_def.returns = None
    func_def.decorator_list = []
    func_def.body = _strip_docstring(func_def.body)

    _RewriteBody().visit(func_def)
    ast.fix_missing_locations(tree)
    result = ast.unparse(func_def)
    params = _build_params(collector)
    body = func_def.body
    is_degenerate = not params and len(body) == 1 and isinstance(body[0], ast.Return)
    return None if is_degenerate else result


def _collect_leaves(skill: Skill) -> list[Skill]:
    """DFS leaves in execution order, skipping grouping nodes and orchestrator fns."""
    if skill.fn and not skill.steps:
        return [skill]
    leaves: list[Skill] = []
    for child in skill.steps:
        leaves.extend(_collect_leaves(child))
    return leaves


def _render_lm(skill: Skill, heading: str) -> str:
    prompt = getattr(skill.fn, "lm_system_prompt", "") or ""
    lines = [f"{heading} {skill.name}"]
    if skill.description:
        lines += ["", skill.description]
    if prompt:
        lines += ["", f"> {prompt}"]
    lines.append("")
    return "\n".join(lines)


def _render_fn(skill: Skill, heading: str) -> str:
    lines = [f"{heading} {skill.name}"]
    if skill.description:
        lines += ["", skill.description]
    src = rewrite_step_source(skill.fn)
    if src is not None:
        lines += ["", "```python", src, "```"]
    lines.append("")
    return "\n".join(lines)


def _skill_preamble(skill: Skill) -> list[str]:
    """Intro block: description and/or orchestrator system prompt."""
    lines: list[str] = []
    desc = skill.description or (inspect.getdoc(skill.fn) if skill.fn else None)
    prompt = getattr(skill.fn, "lm_system_prompt", None) if skill.fn else None
    if desc:
        lines += [desc, ""]
    if prompt:
        lines += [f"> {prompt}", ""]
    return lines


def compile_skill(skill: Skill) -> str:
    """Produce a readable skill.md from a Skill tree."""
    lines = [f"# {skill.name}", ""]
    lines.extend(_skill_preamble(skill))
    for leaf in _collect_leaves(skill):
        if _is_lm(leaf):
            lines.append(_render_lm(leaf, "##"))
        else:
            lines.append(_render_fn(leaf, "##"))
    return "\n".join(lines).rstrip() + "\n"


# %% [markdown]
# ## Example 1 — calendar booking (orchestrator + regex leaves)

# %%
if __name__ == "__main__":
    import json
    import re

    from tk.llmbda import (
        LMCaller,
        SkillContext,
        StepResult,
        last,
        lm,
        run_skill,
        strip_fences,
    )

    WEEKDAYS = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    _TIME_RE = re.compile(
        r"\b(\d{1,2})(?::(\d{2}))?(?:\s*-\s*(\d{1,2})(?::(\d{2}))?)?\s*(am|pm)?\b",
        re.IGNORECASE,
    )
    _DUR_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*(hour|hr|hrs|minute|min|mins)s?\b",
        re.IGNORECASE,
    )
    _TOPIC_RE = re.compile(r"(?:about|re:)\s+(.+?)(?:[.!?]|$)", re.IGNORECASE)

    def parse_weekday(ctx: SkillContext) -> StepResult:
        """Find an explicit weekday name (Monday..Sunday)."""
        text = ctx.entry["text"].lower()
        for day in WEEKDAYS:
            if re.search(rf"\b{day}\b", text):
                return StepResult(value=day.capitalize(), meta={"reason": "matched"})
        return StepResult(meta={"reason": "no_weekday"})

    def _fmt(h: int, m: int, ampm: str | None) -> str:
        return f"{h}:{m:02d}{ampm.lower()}" if ampm else f"{h:02d}:{m:02d}"

    def parse_time(ctx: SkillContext) -> StepResult:
        """Find a clock time like '3pm', '15:00', or a range '9-10am'."""
        match = _TIME_RE.search(ctx.entry["text"])
        if not match:
            return StepResult(meta={"reason": "no_time"})
        h1, min1, h2, min2, ampm = match.groups()
        start = _fmt(int(h1), int(min1 or 0), ampm)
        end = _fmt(int(h2), int(min2 or 0), ampm) if h2 else None
        return StepResult(value={"start": start, "end": end}, meta={"range": bool(end)})

    def parse_duration(ctx: SkillContext) -> StepResult:
        """Find a duration phrase like '30 minutes' or '2 hrs' and return minutes."""
        match = _DUR_RE.search(ctx.entry["text"])
        if not match:
            return StepResult(meta={"reason": "no_duration"})
        n, unit = float(match.group(1)), match.group(2).lower()
        minutes = int(n * 60) if unit.startswith(("hour", "hr")) else int(n)
        return StepResult(value=minutes, meta={"reason": "matched"})

    def parse_topic(ctx: SkillContext) -> StepResult:
        """Find a topic phrase introduced by 'about' or 're:'."""
        match = _TOPIC_RE.search(ctx.entry["text"])
        if not match:
            return StepResult(meta={"reason": "no_topic"})
        return StepResult(value=match.group(1).strip(), meta={"reason": "matched"})

    VERIFY_PROMPT = """\
You are a calendar booking verifier.
Input JSON has "text" (original request) and "prior_steps" (each parser's
name/description/value/meta). Cross-check the prior findings against
the text: confirm, correct, fill gaps (no invention), flag ambiguity.

Return ONLY JSON:
{
  "booking": {"weekday": ..., "start": ..., "end": ..., "minutes": ..., "topic": ...},
  "notes": "<one sentence>"
}
"""
    _CANNED_BOOKINGS = {
        "Thursday at 3pm about Q3 planning": {
            "booking": {
                "weekday": "Thursday",
                "start": "3:00pm",
                "end": None,
                "minutes": None,
                "topic": "Q3 planning",
            },
            "notes": (
                "The request includes weekday, start time, and topic but no duration."
            ),
        }
    }

    def scripted_booking_caller(
        *, messages: list[dict[str, str]], **_kw: object
    ) -> str:
        """Scripted LMCaller for examples; returns a JSON string."""
        user_msg = messages[1]["content"]
        for key, payload in _CANNED_BOOKINGS.items():
            if key in user_msg:
                return json.dumps(payload)
        return json.dumps(
            {"booking": {}, "notes": "No canned response for this input."}
        )

    def _prior_payload(
        trace: dict[str, StepResult],
        skills: list[Skill],
    ) -> list[dict[str, object]]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "value": trace[skill.name].value,
                "meta": trace[skill.name].meta,
            }
            for skill in skills
            if skill.name in trace
        ]

    @lm(scripted_booking_caller, system_prompt=VERIFY_PROMPT)
    def verify(ctx: SkillContext, steps: list[Skill], call: LMCaller) -> StepResult:
        """Run parser children, cross-check extractions against the raw text."""
        inner = Skill(name="_parse", steps=steps)
        trace = run_skill(inner, ctx.entry)
        payload = {
            "text": ctx.entry["text"],
            "prior_steps": _prior_payload(trace, steps),
        }
        raw = call(
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}]
        )
        parsed = json.loads(strip_fences(raw))
        return StepResult(
            value=parsed.get("booking"),
            meta={"notes": parsed.get("notes", ""), "llm_raw": raw},
        )

    book_meeting = Skill(
        name="book_meeting",
        fn=verify,
        steps=[
            Skill("λ::weekday", fn=parse_weekday),
            Skill("λ::time", fn=parse_time),
            Skill("λ::duration", fn=parse_duration),
            Skill("λ::topic", fn=parse_topic),
        ],
    )

    compiled_booking = compile_skill(book_meeting)
    print(compiled_booking)
    assert "# book_meeting" in compiled_booking
    assert "## λ::weekday" in compiled_booking
    assert "def parse_weekday(text):" in compiled_booking
    assert VERIFY_PROMPT.strip() in compiled_booking

    booking_trace = run_skill(
        Skill(name="s", steps=[book_meeting]),
        text="Let's meet Thursday at 3pm about Q3 planning",
    )
    booking_result = last(booking_trace)
    assert isinstance(booking_result.value, dict)
    assert booking_result.value["weekday"] == "Thursday"
    assert booking_result.meta["notes"]
    print("book_meeting result:", booking_result.value)

    print("=" * 60)

    # %% [markdown]
    # ## Example 2 — retry orchestrator with LLM extract + deterministic validate

    # %%
    _ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    _attempt = 0

    def fake_flaky_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        """Return invalid output first, then a valid ISO date."""
        del messages
        global _attempt  # noqa: PLW0603
        _attempt += 1
        return "not-a-date" if _attempt == 1 else "2025-06-01"

    @lm(fake_flaky_model, system_prompt="Extract a date as YYYY-MM-DD.")
    def extract_date_lm(ctx: SkillContext, call: LMCaller) -> StepResult:
        """Extract a date via LLM."""
        raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
        return StepResult(value=raw.strip(), meta={"source": "lm"})

    def validate_iso(ctx: SkillContext) -> StepResult:
        """Check whether the latest extraction is a valid ISO-8601 date."""
        value = ctx.trace["ψ::extract"].value
        if _ISO_RE.fullmatch(str(value or "")):
            return StepResult(value=value, meta={"valid": True})
        return StepResult(meta={"valid": False})

    def retry_extract_verify(ctx: SkillContext, steps: list[Skill]) -> StepResult:
        """Run extract→validate up to 3 times until valid."""
        inner = Skill(name="inner", steps=steps)
        result = StepResult(meta={"valid": False})
        for attempt in range(1, 4):
            trace = run_skill(inner, ctx.entry)
            result = last(trace)
            if result.meta.get("valid"):
                return StepResult(
                    value=result.value,
                    meta={"valid": True, "attempts": attempt},
                )
        return StepResult(
            value=result.value,
            meta={"valid": False, "attempts": 3},
        )

    dates_skill = Skill(
        name="dates",
        fn=retry_extract_verify,
        steps=[
            Skill("ψ::extract", fn=extract_date_lm),
            Skill("λ::validate", fn=validate_iso),
        ],
    )

    compiled_dates = compile_skill(dates_skill)
    print(compiled_dates)
    assert "# dates" in compiled_dates
    assert "## ψ::extract" in compiled_dates
    assert "## λ::validate" in compiled_dates
    assert "value = extract" in compiled_dates

    dates_trace = run_skill(dates_skill, text="Let's meet on the 15th of January 2025")
    dates_result = last(dates_trace)
    assert _ISO_RE.fullmatch(str(dates_result.value)), f"{dates_result.value=}"
    assert dates_result.meta.get("valid") is True
    print(
        "dates result:",
        dates_result.value,
        "attempts:",
        dates_result.meta.get("attempts"),
    )
