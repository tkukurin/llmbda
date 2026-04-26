# tk.llmbda

Skill composition for LLM pipelines. Chain deterministic and LLM-powered
steps into a skill; the runtime walks them in order until one resolves.

## Deterministic skill

```python
from tk.llmbda import Skill, SkillContext, StepResult, run_skill

def greet(ctx: SkillContext) -> StepResult:
    name = ctx.entry.get("name", "world")
    return StepResult(value=f"hello, {name}")

skill = Skill(name="greeter", steps=[Skill("greet", fn=greet)])
result = run_skill(skill, {"name": "λ"})
# SkillResult(skill="greeter", resolved_by="greet", value="hello, λ", ...)
```

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
result = run_skill(skill, {"text": "let's meet on the 15th of January 2025"})
# SkillResult(skill="dates", resolved_by="extract", value="2025-01-15", ...)
```

## Multi-step with `ctx.prev`

Each step can access the previous step's result via `ctx.prev`:

```python
from tk.llmbda import Skill, SkillContext, StepResult, run_skill

def step_a(ctx: SkillContext) -> StepResult:
    return StepResult(value=ctx.entry["x"] * 2)

def step_b(ctx: SkillContext) -> StepResult:
    return StepResult(value=ctx.prev.value + 10)

skill = Skill(name="math", steps=[Skill("a", fn=step_a), Skill("b", fn=step_b)])
result = run_skill(skill, {"x": 5})
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
```

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

A skill with both `fn` and `steps` is an orchestrator — `fn` controls how
the children execute. The runtime sets `ctx.skills` to the skill's children
before calling `fn`, so there's no duplication:

```python
def retry(ctx: SkillContext) -> StepResult:
    """Run children up to 3 times until valid."""
    inner = Skill(name="inner", steps=ctx.skills)
    for attempt in range(1, 4):
        r = run_skill(inner, ctx.entry)
        if r.metadata.get("valid"):
            return StepResult(value=r.value, metadata={"attempts": attempt})
    return StepResult(value=r.value, metadata={"valid": False})

skill = Skill(
    name="retry",
    fn=retry,
    steps=[
        Skill("ψ::extract", fn=extract_date_llm),
        Skill("ψ::verify", fn=verify_date),
    ],
)
```

- Children run in a fresh `SkillContext` (via `run_skill`), so they don't
  see the outer trace. Pass data through `entry` if needed.
- `ctx.skills` is restored after `fn` returns (or raises).

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
- **`ctx.prev`** — most recently executed step's `StepResult`. Starts as `ROOT` (`value=None`).
- **`ctx.trace`** — dict of all prior results keyed by step name. Raises informative `KeyError` on miss; use `.get()` for optional lookups.
- **`ctx.entry`** — the original input passed to `run_skill`.
- **`ctx.skills`** — for orchestrators: the skill's declared children. For leaves: the top-level leaves list.
- **`@lm(model, system_prompt=...)`** — binds model at decoration time. Decorated fn is `(ctx, call)`; `call` prepends `system_prompt`.
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
