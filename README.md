# tk.llmbda

Skill composition for LLM pipelines. Chain deterministic and LLM-powered
steps into a skill; the runtime walks them in order until one resolves.

```python
from tk.llmbda import Skill, Step, StepContext, StepResult, lm, run_skill

def greet(ctx: StepContext) -> StepResult:
    name = ctx.entry.get("name", "world")
    return StepResult(value=f"hello, {name}")

skill = Skill(
    name="greeter",
    steps=[Step("greet", greet)],
)

result = run_skill(skill, {"name": "λ"})
# SkillResult(skill="greeter", resolved_by="greet", value="hello, λ", ...)
```

- **`@lm(model, system_prompt=...)`** — decorator that binds a model (and optional system prompt) to a step function at definition time. The decorated function receives `(ctx, call)` where `call` forwards to the model with the system prompt prepended. The model must exist before decoration.
- **`Step.description`** — auto-populated from the function's docstring via `__post_init__`. Pass an explicit `description=` to override. Later steps can read prior step descriptions from `ctx.steps` for cross-checking context.
- **`StepResult.value`** — the step's actual output: parsed data, extracted values, model responses, or `None` when nothing was found.
- **`StepResult.metadata`** — auxiliary context about the step: reasons, diagnostics, raw provider output, parse errors, confidence, or other non-primary details.
- **`StepResult.resolved`** — defaults to `True`; return `resolved=False` to fall through to the next step. Execution stops after the final step regardless, and the trace preserves the flag that step returned.
- **`ctx.steps`, `ctx.prior`** — the plan and prior-step outcomes, for steps that cross-check or summarise earlier work. When serialising prior steps for an LLM, include both `value` and `metadata`.
- **`iter_skill`** — same execution as `run_skill`, but yields `(step_name, result)` for live observation or early exit.

## LLM steps

```python
from tk.llmbda import lm, StepContext, StepResult

@lm(my_model, system_prompt="Extract the date from the text.")
def extract_date(ctx: StepContext, call) -> StepResult:
    """Extract a date from natural language text."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(value=raw)
```

- The `call` argument is a bound caller that prepends the system prompt to `messages` before forwarding to the model.
- Introspection: `extract_date.lm_system_prompt`, `extract_date.lm_model`.
- Re-bind for testing via `__wrapped__`: `lm(fake)(extract_date.__wrapped__)`.