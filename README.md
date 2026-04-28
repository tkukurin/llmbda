# tk.llmbda

Skill composition for LLM pipelines. Chain deterministic and LLM-powered
steps into a skill; the runtime walks them in order until one exits.

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

```python
from litellm import completion
from tk.llmbda import Skill, SkillContext, StepResult, lm, run_skill

def call_lm(*, messages, **kw):
    resp = completion(model="gpt-4o-mini", messages=messages, **kw)
    return resp.choices[0].message.content

@lm(call_lm, system_prompt="Extract a date. Return ISO format.")
def extract_date(ctx: SkillContext, call) -> StepResult:
    """Extract a date from natural language."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(value=raw.strip())

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

## Short-circuit with `exits`

Steps fall through by default (`exits=()`). Set `exits=True` to
stop early — remaining steps are skipped:

```python
def try_cache(ctx: SkillContext) -> StepResult:
    if ctx.entry.get("key") in CACHE:
        return StepResult(value=CACHE[ctx.entry["key"]], exits=True)
    return StepResult(value=None)

def expensive(ctx: SkillContext) -> StepResult:
    return StepResult(value=compute(ctx.entry))

skill = Skill(name="s", steps=[Skill("cache", fn=try_cache), Skill("compute", fn=expensive)])
# run_skill(skill, key="known-key")
```

Use explicit `StepResult` when you need `exits` or `metadata`.
Orchestrators can pass a tuple for provenance: `exits=("child_name",)`.

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
            return StepResult(value=r.value, metadata={"attempts": attempt}, exits=r.resolved_by)
    return StepResult(value=r.value, metadata={"valid": False}, exits=r.resolved_by)

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

## `ctx.trace` — cross-step access by name

```python
from tk.llmbda import Skill, SkillContext, run_skill

def extract(ctx: SkillContext) -> str:
    return ctx.entry["text"].upper()

def summarize(ctx: SkillContext) -> str:
    return f"extracted: {ctx.trace['extract'].value}"

skill = Skill(name="pipe", steps=[
    Skill("extract", fn=extract),
    Skill("summarize", fn=summarize),
])
result = run_skill(skill, text="hello")
# result.value == "extracted: HELLO"
```

Use `ctx.trace.get("key")` for optional lookups; missing keys raise an
informative `KeyError`.

## `iter_skill` — streaming / early exit

```python
from tk.llmbda import Skill, iter_skill

skill = Skill(name="s", steps=[step_a, step_b, step_c])
for name, result in iter_skill(skill, {"x": 1}):
    print(name, result.value)
    if result.exits:
        break
```

## Test re-binding

```python
from tk.llmbda import lm

fake_model = lambda *, messages, **kw: "2025-01-15"
testable = lm(fake_model)(extract_date.__wrapped__)
```

## Examples

```bash
# deterministic + LLM date extraction with retry (uses litellm)
uv run examples/date_extraction.py

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
