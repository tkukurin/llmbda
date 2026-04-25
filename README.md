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

- **`caller`** — OpenAI-compatible `Callable[..., str]`. Required; pass a noop for deterministic-only skills.
- **`Step.system_prompt`** — the runtime auto-prepends it as a system message on any `ctx.caller(messages=...)` call inside the step. Don't add a system message yourself.
- **`StepResult.resolved`** — defaults to `True`; return `resolved=False` to fall through to the next step. The last step is always treated as resolved.
- **`ctx.steps`, `ctx.prior`** — the plan and prior-step outcomes, for steps that cross-check or summarise earlier work.
- **`iter_skill`** — same execution as `run_skill`, but yields `(step_name, result)` for live observation or early exit.