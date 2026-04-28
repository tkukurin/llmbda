# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm>=1.0",
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
    return skill.fn is not None and bool(getattr(skill.fn, "lm_system_prompt", None))


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
            node.attr in ("value", "metadata", "resolved")
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

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        # ctx.trace["k"].value / .metadata  →  k / k_metadata
        if node.attr in ("value", "metadata"):
            key = self._trace_key(node.value)
            if key is not None:
                name = _sanitize(key)
                if node.attr != "value":
                    name = f"{name}_{node.attr}"
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        # ctx.prev.value  →  prev_value
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "ctx"
            and node.value.attr == "prev"
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
            meta = args[1] if len(args) > 1 else kw.get("metadata")
            if meta:
                node.value = ast.Dict(
                    keys=[ast.Constant(value="value"), ast.Constant(value="metadata")],
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


def rewrite_step_source(fn: Any) -> str:
    """Rewrite a step fn's source to remove framework boilerplate.

    - Replaces ctx.entry["key"] with a direct *key* parameter.
    - Replaces ctx.trace["name"].value with a *name* parameter.
    - Replaces ctx.prev.value with a *prev_value* parameter.
    - Unwraps StepResult(value, ...) returns to bare value returns.
    - Strips the docstring (shown separately), decorators, and return annotation.
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

    from litellm import completion

    from tk.llmbda import (
        LMCaller,
        SkillContext,
        StepResult,
        lm,
        run_skill,
        strip_fences,
    )

    MODEL = "gpt-4o-mini"

    def call_lm(*, messages: list[dict[str, str]], **kw: Any) -> str:
        resp = completion(model=MODEL, messages=messages, **kw)
        return resp.choices[0].message.content

    WEEKDAYS = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )

    def parse_weekday(ctx: SkillContext) -> StepResult:
        """Find an explicit weekday name (Monday..Sunday)."""
        text = ctx.entry["text"].lower()
        for day in WEEKDAYS:
            if re.search(rf"\b{day}\b", text):
                return StepResult(day.capitalize(), {"reason": "matched"})
        return StepResult(None, {"reason": "no_weekday"})

    def parse_time(_ctx: SkillContext) -> StepResult:
        """Find a clock time like '3pm', '15:00', or a range '9-10am'."""
        return StepResult(None, {"reason": "no_time"})

    def parse_duration(_ctx: SkillContext) -> StepResult:
        """Find a duration phrase like '30 minutes' or '2 hrs'."""
        return StepResult(None, {"reason": "no_duration"})

    def parse_topic(_ctx: SkillContext) -> StepResult:
        """Find a topic phrase introduced by 'about' or 're:'."""
        return StepResult(None, {"reason": "no_topic"})

    VERIFY_PROMPT = (
        "You are a calendar booking verifier. Cross-check the prior "
        "findings against the text: confirm, correct, fill gaps, "
        "flag ambiguity. Respond with a JSON object."
    )

    @lm(call_lm, system_prompt=VERIFY_PROMPT)
    def verify(ctx: SkillContext, steps: list[Skill], call: LMCaller) -> StepResult:
        """Cross-check parser extractions against the raw text."""
        inner = Skill(name="_parse", steps=steps)
        r = run_skill(inner, ctx.entry)
        payload = json.dumps(
            {
                "text": ctx.entry["text"],
                "prior_steps": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "value": r.trace[s.name].value,
                    }
                    for s in steps
                    if s.name in r.trace
                ],
            }
        )
        raw = call(
            messages=[{"role": "user", "content": payload}],
            response_format={"type": "json_object"},
        )
        return StepResult(json.loads(strip_fences(raw)))

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

    print(compile_skill(book_meeting))

    result = run_skill(
        book_meeting, text="Let's meet Thursday at 3pm about Q3 planning"
    )
    assert isinstance(result.value, dict)  # LLM verify returns parsed JSON
    assert "book_meeting" in result.trace  # orchestrator is a single trace entry
    print("book_meeting result:", result.value)

    print("=" * 60)

    # %% [markdown]
    # ## Example 2 — retry orchestrator with LLM extract + deterministic validate

    # %%
    _ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

    @lm(call_lm, system_prompt="Extract a date from the text. Return ONLY ISO-8601.")
    def extract_date_lm(ctx: SkillContext, call: Any) -> StepResult:
        """Extract a date via LLM."""
        raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
        return StepResult(raw.strip(), {"source": "lm"})

    def validate_iso(ctx: SkillContext) -> StepResult:
        """Check whether the latest extraction is a valid ISO-8601 date."""
        prev = ctx.trace.get("ψ::extract")
        if prev and _ISO_RE.fullmatch(str(prev.value or "")):
            return StepResult(prev.value, {"valid": True})
        return StepResult(None, {"valid": False})

    def retry_extract_verify(ctx: SkillContext, steps: list[Skill]) -> StepResult:
        """Run extract→validate up to 3 times until valid."""
        inner = Skill(name="inner", steps=steps)
        for attempt in range(1, 4):
            r = run_skill(inner, ctx.entry)
            if r.metadata.get("valid"):
                return StepResult(
                    value=r.value,
                    metadata={"valid": True, "attempts": attempt},
                    resolved_by=r.resolved_by,
                )
        return StepResult(
            value=r.value,
            metadata={"valid": False, "attempts": 3},
            resolved_by=r.resolved_by,
        )

    dates_skill = Skill(
        name="dates",
        fn=retry_extract_verify,
        steps=[
            Skill("ψ::extract", fn=extract_date_lm),
            Skill("λ::validate", fn=validate_iso),
        ],
    )

    print(compile_skill(dates_skill))

    result = run_skill(dates_skill, text="Let's meet on the 15th of January 2025")
    assert _ISO_RE.fullmatch(str(result.value)), f"{result.value=}"
    assert result.metadata.get("valid") is True  # deterministic validate confirms
    print("dates result:", result.value, "attempts:", result.metadata.get("attempts"))
