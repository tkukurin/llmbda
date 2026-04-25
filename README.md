# tk.llmbda

Proto skill-composition framework for LLM pipelines.
Define deterministic and LLM-powered steps, chain them into skills.

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

`caller` is required. Skills without LLM steps pass a noop; real skills pass
an OpenAI-compatible callable.

Each `Step` carries an optional `system_prompt` describing its intent. LLM
steps forward it to the model; deterministic steps expose it so later LLM
steps can include it as context via `ctx.steps` and `ctx.prior`.

`StepResult.resolved` defaults to `True`. Return `resolved=False` to fall through
to the next step. The last step is always treated as resolved, so the skill is
guaranteed to return something.

`SkillResult.trace` is an ordered `{step_name: StepResult}` of every step that ran.
For live observation or early exit, iterate with `iter_skill` instead:

```python
for name, step_result in iter_skill(skill, entry, caller):
    if step_result.resolved:
        break
```
