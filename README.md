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
# SkillResult(skill="greeter", resolved_by=("greet",), value="hello, λ", ...)
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
# SkillResult(skill="dates", resolved_by=("extract",), value="2025-01-15", ...)
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

## Async execution

`arun_skill` and `aiter_skill` are async equivalents of `run_skill` and
`iter_skill`. Async step functions (`async def`) are awaited automatically:

```python
import asyncio
from tk.llmbda import Skill, SkillContext, StepResult, arun_skill

async def fetch(ctx: SkillContext) -> StepResult:
    await asyncio.sleep(0.01)  # e.g. async HTTP call
    return StepResult(value="fetched")

skill = Skill(name="s", steps=[Skill("fetch", fn=fetch)])
result = asyncio.run(arun_skill(skill, {}))
# result.value == "fetched"
```

The sync runner (`run_skill`) also handles async steps via `asyncio.run`,
but cannot be called from inside a running event loop — use `arun_skill` there.

## Token streaming

Step functions that are generators (sync or async) yield intermediate string
chunks before a final `StepResult`. Use `iter_skill` or `aiter_skill` to
consume them:

```python
from tk.llmbda import Skill, SkillContext, StepResult, iter_skill

def stream(ctx: SkillContext):
    yield "chunk 1 "
    yield "chunk 2"
    yield StepResult(value="done")

skill = Skill("s", steps=[Skill("a", fn=stream)])
for name, item in iter_skill(skill, {}):
    print(name, item)
# a chunk 1
# a chunk 2
# a StepResult(value='done', ...)
```

`run_skill` / `arun_skill` skip string chunks and return only the final result.

## Parallel steps

Set `parallel=True` on a composite `Skill` to run its children concurrently
(async) or sequentially-but-isolated (sync). Each child gets a snapshot of
the parent trace — children see prior steps but not each other's results:

```python
from tk.llmbda import Skill, SkillContext, StepResult, run_skill

def a(ctx: SkillContext) -> StepResult:
    return StepResult(value="A")

def b(ctx: SkillContext) -> StepResult:
    return StepResult(value="B")

skill = Skill("root", steps=[
    Skill("par", parallel=True, steps=[
        Skill("a", fn=a),
        Skill("b", fn=b),
    ])
])
result = run_skill(skill, {})
# result.trace has both "a" and "b"
```

## Concepts

- **`Skill`** — recursive composition primitive. Leaf (`fn`), composite (`steps`), or orchestrator (`fn` + `steps`). Set `parallel=True` on composites for concurrent execution.
- **`StepResult.resolved`** — defaults to `False`; steps fall through. Set `True` to short-circuit.
- **`StepResult.resolved_by`** — inner resolution path as `tuple[str, ...]`. Orchestrators propagate it from nested `run_skill` calls; `SkillResult.resolved_by` prepends the step name, building a hierarchical path like `("orchestrator", "inner_step")`.
- **`ctx.prev`** — most recently executed step's `StepResult`. Starts as `ROOT` (`value=None`).
- **`ctx.trace`** — dict of all prior results keyed by step name. Raises informative `KeyError` on miss; use `.get()` for optional lookups.
- **`ctx.entry`** — the original input passed to `run_skill`.
- **`@lm(model, system_prompt=...)`** — binds model at decoration time. Decorated fn is `(ctx, call)` for leaves or `(ctx, steps, call)` for orchestrators; `call` prepends `system_prompt`.
- **`Skill.description`** — human-readable summary; falls back to fn docstring. Separate from `@lm` system prompts (read those via `fn.lm_system_prompt`).
- **`StepResult.metadata`** — auxiliary context: reasons, raw provider output, confidence.
- **`run_skill` / `arun_skill`** — run to completion, return `SkillResult`. Async variant for use inside event loops.
- **`iter_skill` / `aiter_skill`** — yield `(name, str_chunk | StepResult)` per step. Enables token streaming and early exit.
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
