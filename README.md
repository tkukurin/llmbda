# tk.llmbda

Proto skill-composition framework for LLM pipelines.
Define deterministic and LLM-powered steps, chain them into skills,
compose skills into pipelines.

```python
from tk.llmbda import Skill, Step, StepContext, StepResult, run_skill

def greet(ctx: StepContext) -> StepResult:
    name = ctx.entry.get("name", "world")
    return StepResult(value=f"hello, {name}", terminal=True)

skill = Skill(
    name="greeter",
    steps=[Step("greet", greet)],
)

result = run_skill(skill, {"name": "\u03bb"})
# {"value": "hello, \u03bb", "metadata": {"skill": "greeter", "resolved_by": "greet"}}
```
