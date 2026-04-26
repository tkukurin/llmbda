# tk.llmbda

Skill composition for LLM pipelines. Chain deterministic and LLM-powered
steps into a skill; the runtime walks them in order until one resolves.

```python
from tk.llmbda import Skill, Step, StepContext, StepResult, run_skill

def greet(ctx: StepContext) -> StepResult:
    name = ctx.entry.get("name", "world")
    return StepResult(value=f"hello, {name}")

skill = Skill(
    name="greeter",
    steps=[Step("greet", greet, system_prompt="Greet the user by name.")],
)

result = run_skill(skill, {"name": "\u03bb"}, caller=lambda **_: None)
# SkillResult(skill="greeter", resolved_by="greet", value="hello, \u03bb", ...)
```

- **`caller`** — OpenAI-style keyword callable. Steps usually call it as `ctx.caller(messages=[...])`; it may return any provider-specific value the step knows how to handle. Required; pass a noop for deterministic-only skills.
- **`Step.system_prompt`** — when non-empty, the runtime binds `ctx.caller` so calls with `messages=[...]` receive this prompt as the first system message. Don't add the same system message yourself.
- **`StepResult.value`** — the step's actual output: parsed data, extracted values, model responses, or `None` when nothing was found.
- **`StepResult.metadata`** — auxiliary context about the step: reasons, diagnostics, raw provider output, parse errors, confidence, or other non-primary details.
- **`StepResult.resolved`** — defaults to `True`; return `resolved=False` to fall through to the next step. Execution stops after the final step regardless, and the trace preserves the flag that step returned.
- **`ctx.steps`, `ctx.prior`** — the plan and prior-step outcomes, for steps that cross-check or summarise earlier work. When serialising prior steps for an LLM, include both `value` and `metadata`.
- **`iter_skill`** — same execution as `run_skill`, but yields `(step_name, result)` for live observation or early exit.
