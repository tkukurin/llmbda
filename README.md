# tk.llmbda

> Like lambda calculus, but the functions talk back.

A minimal skill-composition framework for LLM pipelines.
Define deterministic and LLM-powered steps, chain them into skills,
compose skills into pipelines.  Zero dependencies.

## Install

```sh
uv add tk-llmbda
```

## Quick start

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

## Concepts

| Thing | What it does |
|---|---|
| **Step** | A named function: `StepContext -> StepResult` |
| **Skill** | A named sequence of steps + a system prompt |
| **run_skill** | Executes steps in order; first `terminal` result wins |
| **compose** | Concatenates system prompts from multiple skills |
| **Caller** | Protocol for LLM calls -- swap real clients for test fixtures |

## Design principles

- **Zero dependencies.** The core is pure Python dataclasses.
- **Caller is a protocol.** Bring your own LLM client, or use a `lambda` for tests.
- **Steps are just functions.** No decorators, no registration, no magic.
- **Terminal short-circuits.** A deterministic step can bail out before the LLM is ever called.
- **Config-as-code.** Skills are Python objects -- compose them with plain Python, not YAML.

## The name

`llmbda` = LLM + lambda.  The module lives under the `tk` namespace
because every good toolkit deserves one.
