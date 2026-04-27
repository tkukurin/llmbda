# tk.llmbda

Skill composition for LLM pipelines. Chain deterministic and LLM-powered
steps into a skill; the runtime walks them in order until one resolves.

## Deterministic skill

```python
from tk.llmbda import Skill, SkillContext, run_skill

def greet(ctx: SkillContext) -> str:
    return f"hello, {ctx.entry.get('name', 'world')}"

skill = Skill(name="greeter", steps=[greet])
result = run_skill(skill, name="λ")
# SkillResult(skill="greeter", resolved_by=("greet",), value="hello, λ", ...)
```

Step fns that return a non-`StepResult` are auto-wrapped as
`StepResult(value=x)`. Bare callables in `steps` are auto-wrapped as
`Skill(name=fn.__name__, fn=fn)`. Keyword arguments to `run_skill` become
the `ctx.entry` dict.

## LLM skill

Self-contained with a fake model so the snippet runs as-is:

```python
from tk.llmbda import Skill, SkillContext, StepResult, lm, run_skill

def fake_model(*, messages, **_):
    return "2025-01-15"

@lm(fake_model, system_prompt="Extract a date. Return ISO format.")
def extract_date(ctx: SkillContext, call) -> StepResult:
    """Extract a date from natural language."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(value=raw)

skill = Skill(name="dates", steps=[Skill("extract", fn=extract_date)])
result = run_skill(skill, text="let's meet on the 15th of January 2025")
# SkillResult(skill="dates", resolved_by=("extract",), value="2025-01-15", ...)
```

## Multi-step with `ctx.prev`

Each step can access the previous step's result via `ctx.prev`:

```python
from tk.llmbda import Skill, SkillContext, run_skill

def double(ctx: SkillContext) -> int:
    return ctx.entry["x"] * 2

def add_ten(ctx: SkillContext) -> int:
    return ctx.prev.value + 10

skill = Skill(name="math", steps=[double, add_ten])
result = run_skill(skill, x=5)
# result.value == 20
```

Before any step runs, `ctx.prev` is the `ROOT` sentinel (`ROOT.value is None`).
Use `ctx.prev is ROOT` to check if you're the first step.

## Short-circuit with `resolved=True`

Steps fall through by default (`resolved=False`). Set `resolved=True` to
stop early — remaining steps are skipped:

```python
def try_cache(ctx: SkillContext) -> StepResult:
    if ctx.entry.get("key") in CACHE:
        return StepResult(value=CACHE[ctx.entry["key"]], resolved=True)
    return StepResult(value=None)

def expensive(ctx: SkillContext) -> StepResult:
    return StepResult(value=compute(ctx.entry))

skill = Skill(name="s", steps=[Skill("cache", fn=try_cache), Skill("compute", fn=expensive)])
# run_skill(skill, key="known-key")
```

Use explicit `StepResult` when you need `resolved`, `metadata`, or `resolved_by`.

## Nested composition

`Skill` is the single composition primitive. Leaf (has `fn`), composite
(has `steps`), or orchestrator (has both):

```python
skill = Skill(
    name="analyzer",
    steps=[
        Skill("preprocess", steps=[
            Skill("normalize", fn=normalize),
            Skill("count_words", fn=count_words),
        ]),
        Skill("classify", steps=[
            Skill("tag_length", fn=tag_length),
        ]),
    ],
)
```

The runtime walks composites via DFS. All leaves share a single `ctx.trace`.

## Orchestrator: `fn` + `steps`

A skill with both `fn` and `steps` is an orchestrator — `fn` receives the
children as an explicit second argument and controls how they execute:

```python
def retry(ctx: SkillContext, steps: list[Skill]) -> StepResult:
    """Run children up to 3 times until valid."""
    inner = Skill(name="inner", steps=steps)
    for attempt in range(1, 4):
        r = run_skill(inner, ctx.entry)
        if r.metadata.get("valid"):
            return StepResult(value=r.value, metadata={"attempts": attempt}, resolved_by=r.resolved_by)
    return StepResult(value=r.value, metadata={"valid": False}, resolved_by=r.resolved_by)

skill = Skill(
    name="retry",
    fn=retry,
    steps=[
        Skill("ψ::extract", fn=extract_date_llm),
        Skill("ψ::verify", fn=verify_date),
    ],
)
```

- Leaf fns are `(ctx) -> StepResult`. Orchestrator fns are `(ctx, steps) -> StepResult`.
- Children run in a fresh `SkillContext` (via `run_skill`), so they don't
  see the outer trace. Pass data through `entry` if needed.

## Static validation

`check_skill` catches trace-key typos and forward references at definition
time via AST analysis:

```python
from tk.llmbda import check_skill

issues = check_skill(skill)
# ["'bad_step' references undeclared trace key 'typo'"]
```

- Validates orchestrator children as a separate scope.
- Checks `ctx.trace["key"]` and `ctx.trace.get("key")` patterns.

## Concepts

- **`Skill`** — recursive composition primitive. Leaf (`fn`), composite (`steps`), or orchestrator (`fn` + `steps`).
- **`StepResult.resolved`** — defaults to `False`; steps fall through. Set `True` to short-circuit.
- **`StepResult.resolved_by`** — inner resolution path as `tuple[str, ...]`. Orchestrators propagate it from nested `run_skill` calls; `SkillResult.resolved_by` prepends the step name, building a hierarchical path like `("orchestrator", "inner_step")`.
- **Implicit `StepResult`** — step fns can return any value; non-`StepResult` returns are wrapped as `StepResult(value=x)`. Use explicit `StepResult` for `resolved`, `metadata`, or `resolved_by`.
- **Bare callables in `steps`** — `steps=[my_fn]` auto-wraps to `Skill(name=fn.__name__, fn=my_fn)`. Use explicit `Skill(name, fn=...)` for custom names.
- **`run_skill` / `iter_skill` kwargs** — `run_skill(skill, name="λ")` is sugar for `run_skill(skill, {"name": "λ"})`.
- **`ctx.prev`** — most recently executed step's `StepResult`. Starts as `ROOT` (`value=None`).
- **`ctx.trace`** — dict of all prior results keyed by step name. Raises informative `KeyError` on miss; use `.get()` for optional lookups.
- **`ctx.entry`** — the original input passed to `run_skill`.
- **`@lm(model, system_prompt=...)`** — binds model at decoration time. Decorated fn is `(ctx, call)` for leaves or `(ctx, steps, call)` for orchestrators; `call` prepends `system_prompt`.
- **`Skill.description`** — human-readable summary; falls back to fn docstring. Separate from `@lm` system prompts (read those via `fn.lm_system_prompt`).
- **`StepResult.metadata`** — auxiliary context: reasons, raw provider output, confidence.
- **`iter_skill`** — same as `run_skill` but yields `(name, result)` per step for streaming or early exit.
- **`check_skill`** — static validation of trace-key references. Catches typos and forward refs.
- **Test re-binding** — `lm(fake)(my_step.__wrapped__)` re-decorates with a different model.

## Examples

```bash
# deterministic + LLM date extraction with retry
OPENAI_API_KEY=sk-... uv run examples/readme.py

# calendar booking: regex parsers + LLM verifier
uv run examples/calendar_booking.py

# support triage: extraction, classification, validation loop
uv run examples/support_triage.py

# all 20 use cases in one file (no external deps)
uv run examples/showcase.py
```

## Development

```bash
# activate the pre-push hook (runs ruff + pytest before each push)
git config core.hooksPath .githooks

# skip the hook when you need to force-push a WIP
git push --no-verify
```
