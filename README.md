# tk.llmbda

Skill composition for LLM pipelines. Chain deterministic and LLM-powered
steps into a skill; the runtime walks them in order until one resolves.

## Deterministic skill

```python
from tk.llmbda import Skill, Step, StepContext, StepResult, lm, run_skill

def greet(ctx: StepContext) -> StepResult:
    name = ctx.entry.get("name", "world")
    return StepResult(value=f"hello, {name}")

skill = Skill(name="greeter", steps=[Step("greet", greet)])
result = run_skill(skill, {"name": "λ"})
# SkillResult(skill="greeter", resolved_by="greet", value="hello, λ", ...)
```

## LLM skill

Self-contained with a fake model so the snippet runs as-is:

```python
from tk.llmbda import Skill, Step, StepContext, StepResult, lm, run_skill

def fake_model(*, messages, **_):
    return "2025-01-15"  # pretend the LLM returned an ISO date

@lm(fake_model, system_prompt="Extract a date. Return ISO format.")
def extract_date(ctx: StepContext, call) -> StepResult:
    """Extract a date from natural language."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(value=raw)

skill = Skill(name="dates", steps=[Step("extract_date", extract_date)])
result = run_skill(skill, {"text": "let's meet on the 15th of January 2025"})
# SkillResult(skill="dates", resolved_by="extract_date", value="2025-01-15", ...)
```

## Loop

`loop(*steps, name=..., max_iter=..., until=...)` repeats inner steps until
`until(ctx)` returns `True`, an inner step returns `resolved=True`, or
`max_iter` is hit. It returns a single `Step`, so it composes anywhere a
step can go:

```python
from tk.llmbda import Skill, Step, StepContext, StepResult, loop, run_skill

def draft(ctx: StepContext) -> StepResult:
    prev = ctx.prior.get("draft")
    v = (prev.value + 1) if prev else 1
    return StepResult(value=v, resolved=False)

def check(ctx: StepContext) -> StepResult:
    return StepResult(value=ctx.prior["draft"].value, resolved=False)

skill = Skill(
    name="refine",
    steps=[
        loop(
            Step("draft", draft),
            Step("check", check),
            name="refine_loop",
            max_iter=5,
            until=lambda ctx: ctx.prior["draft"].value >= 3,
        ),
    ],
)
result = run_skill(skill, {})
# SkillResult(skill="refine", resolved_by="refine_loop", value=3, ...)
```

- `max_iter` exhaustion returns `resolved=False` (falls through to next step).
- `until` satisfied or inner `resolved=True` returns `resolved=True`.
- Inner steps update `ctx.prior` each iteration, so they see each other's latest output.
- Loops nest: a `loop(...)` inside another `loop(...)` works as expected.

See `examples/readme.py` for a loop that validates and retries LLM date extraction:

```bash
OPENAI_API_KEY=sk-... uv run examples/readme.py
```

## Concepts

- **`@lm(model, system_prompt=...)`** — binds *model* (and optional system prompt) at decoration time. Decorated fn signature is `(ctx, call)`; `call` prepends `system_prompt` before forwarding to *model*.
- **`Step.description`** — human-readable summary; falls back to the fn docstring via `__post_init__`. Separate from `@lm` system prompts; read those via `step.fn.lm_system_prompt`.
- **`StepResult.value`** — the step's output: parsed data, extracted values, model responses, or `None`.
- **`StepResult.metadata`** — auxiliary context: reasons, raw provider output, parse errors, confidence.
- **`StepResult.resolved`** — defaults to `True`; return `resolved=False` to fall through. Execution stops after the final step regardless.
- **`ctx.steps`, `ctx.prior`** — the plan and prior-step outcomes. Serialise both `value` and `metadata` when passing to a later LLM step.
- **`loop(*steps, name, max_iter, until)`** — repeats inner steps, returns a single `Step`. Stops on `until(ctx)`, inner `resolved=True`, or `max_iter`. Exhaustion returns `resolved=False`.
- **`iter_skill`** — same execution as `run_skill` but yields `(name, result)` per step for live observation or early exit.
- **Test re-binding** — `lm(fake)(my_step.__wrapped__)` re-decorates the original fn body with a different model.
