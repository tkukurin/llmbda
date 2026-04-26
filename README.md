# tk.llmbda

Skill composition for LLM pipelines. Chain deterministic and LLM-powered
steps into a skill; the runtime walks them in order until one resolves.

## Deterministic skill

```python
from tk.llmbda import Skill, Step, StepContext, StepResult, run_skill

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

## OpenAI adapter

Any callable matching `LMCaller` (`(*, messages: list[dict], **kwargs) -> str`)
works as the `@lm` model. Minimal adapter using the `openai` SDK:

```python
from openai import OpenAI
client = OpenAI()

def openai_caller(*, messages, **kwargs):
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, **kwargs,
    )
    return resp.choices[0].message.content
```

Drop-in replacement for `fake_model` in the snippet above.

## Concepts

- **`@lm(model, system_prompt=...)`** — binds *model* (and optional system prompt) at decoration time. Decorated fn signature is `(ctx, call)`; `call` prepends `system_prompt` before forwarding to *model*.
- **`Step.description`** — human-readable summary; falls back to the fn docstring via `__post_init__`. Separate from `@lm` system prompts; read those via `step.fn.lm_system_prompt`.
- **`StepResult.value`** — the step's output: parsed data, extracted values, model responses, or `None`.
- **`StepResult.metadata`** — auxiliary context: reasons, raw provider output, parse errors, confidence.
- **`StepResult.resolved`** — defaults to `True`; return `resolved=False` to fall through. Execution stops after the final step regardless.
- **`ctx.steps`, `ctx.prior`** — the plan and prior-step outcomes. Serialise both `value` and `metadata` when passing to a later LLM step.
- **`iter_skill`** — same execution as `run_skill` but yields `(name, result)` per step for live observation or early exit.
- **Test re-binding** — `lm(fake)(my_step.__wrapped__)` re-decorates the original fn body with a different model.