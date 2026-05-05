# tk.llmbda

Skill composition for LLM pipelines. Chain deterministic and LLM-powered
steps into a skill; the runtime walks them in order and returns an ordered
trace dict.

## Deterministic skill

```python
from tk.llmbda import Skill, SkillContext, last, run_skill

def greet(ctx: SkillContext) -> str:
    return f"hello, {ctx.entry.get('name', 'world')}"

skill = Skill(name="greeter", steps=[greet])
trace = run_skill(skill, name="λ")
# last(trace).value == "hello, λ"
```

Step fns that return a non-`StepResult` are auto-wrapped as
`StepResult(value=x)`. Bare callables in `steps` are auto-wrapped as
`Skill(name=fn.__name__, fn=fn)`. Keyword arguments to `run_skill` become
the `ctx.entry` dict.

## LLM skill

```python
from litellm import completion
from tk.llmbda import Skill, SkillContext, StepResult, last, lm, run_skill

def call_lm(*, messages, **kw):
    resp = completion(model="gpt-4o-mini", messages=messages, **kw)
    return resp.choices[0].message.content

@lm(call_lm, system_prompt="Extract a date. Return ISO format.")
def extract_date(ctx: SkillContext, call) -> StepResult:
    """Extract a date from natural language."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(value=raw.strip())

skill = Skill(name="dates", steps=[Skill("extract", fn=extract_date)])
trace = run_skill(skill, text="let's meet on the 15th of January 2025")
# last(trace).value == "2025-01-15"
```

## Multi-step with `ctx.prev`

Each step can access the previous step's result via `ctx.prev`:

```python
from tk.llmbda import Skill, SkillContext, last, run_skill

def double(ctx: SkillContext) -> int:
    return ctx.entry["x"] * 2

def add_ten(ctx: SkillContext) -> int:
    return ctx.prev.value + 10

skill = Skill(name="math", steps=[double, add_ten])
trace = run_skill(skill, x=5)
# last(trace).value == 20
```

Before any step runs, `ctx.prev` is an empty `StepResult()` (i.e.
`ctx.prev.value is None`).

## `run_skill` returns a `Trace`

`run_skill` returns an ordered dict mapping step names to their
`StepResult`. Use `last(trace)` to get the final step's result:

```python
trace = run_skill(skill, x=5)
trace["double"].value   # 10
trace["add_ten"].value  # 20
last(trace).value       # 20
```

## Control flow via orchestrators

Steps always fall through by default — every step in a flat pipeline runs.
For early-exit, retry, or branching, use an orchestrator: a skill with
both `fn` and `steps`. The `fn` receives children as a second argument and
controls how they execute.

### `fst_match` — built-in first-non-None orchestrator

```python
from tk.llmbda import Skill, fst_match, last, run_skill

skill = Skill(
    name="cached",
    fn=fst_match,
    steps=[Skill("cache", fn=try_cache), Skill("compute", fn=expensive)],
)
trace = run_skill(Skill(name="s", steps=[skill]), key="known-key")
# last(trace).value == "cached-value"
```

### Custom orchestrators

Retry pattern:

```python
def retry(ctx: SkillContext, steps: list[Skill]) -> StepResult:
    """Run children up to 3 times until valid."""
    inner = Skill(name="_", steps=steps)
    for attempt in range(1, 4):
        r = run_skill(inner, ctx.entry)
        v = last(r)
        if v.meta.get("valid"):
            return StepResult(value=v.value, meta={"attempts": attempt})
    return StepResult(value=v.value, meta={"valid": False, "attempts": 3})
```

- Leaf fns are `(ctx) -> StepResult`. Orchestrator fns are `(ctx, steps) -> StepResult`.
- Children run in a fresh `SkillContext` (via `run_skill`), so they don't
  see the outer trace. Pass data through `entry` if needed.

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
trace = run_skill(skill, text="hello")
# last(trace).value == "extracted: HELLO"
```

Use `ctx.trace.get("key")` for optional lookups; missing keys raise an
informative `KeyError`.

## `iter_skill` — streaming / early break

```python
from tk.llmbda import Skill, iter_skill

skill = Skill(name="s", steps=[step_a, step_b, step_c])
for name, result in iter_skill(skill, {"x": 1}):
    print(name, result.value)
    if some_condition(result):
        break
```

## Async API

`arun_skill`, `aiter_skill`, and `afst_match` are async equivalents.
They handle both sync and async step fns — async fns are awaited, sync
fns are called inline.

```python
from tk.llmbda import Skill, SkillContext, StepResult, arun_skill, lm

async def my_model(*, messages, **kw):
    ...  # any async model client

@lm(my_model, system_prompt="Extract a date.")
async def extract(ctx: SkillContext, call) -> StepResult:
    raw = await call(messages=[{"role": "user", "content": ctx.entry}])
    return StepResult(value=raw.strip())

skill = Skill(name="s", steps=[Skill("extract", fn=extract)])
trace = await arun_skill(skill, "meet on Jan 15 2025")
```

- `@lm` detects `async def` and produces an async wrapper automatically.
- `arun_skill` works with mixed sync/async steps in the same pipeline.

## Test re-binding

```python
from tk.llmbda import lm

fake_model = lambda *, messages, **kw: "2025-01-15"
testable = lm(fake_model)(extract_date.__wrapped__)
```

## Examples

All experiments run from a single entrypoint:

```bash
# quick demo (runs skill on a sample input)
uv run examples/__main__.py crag
uv run examples/__main__.py gsm8k
uv run examples/__main__.py triage

# override model
uv run examples/__main__.py crag --model openai/gpt-4o
LLMBDA_MODEL=anthropic/claude-sonnet-4-20250514 uv run examples/__main__.py gsm8k

# full Inspect AI evaluation
uv run examples/__main__.py gsm8k --score --limit 50
uv run examples/__main__.py crag --score
INSPECT_MODEL=none/none uv run examples/__main__.py triage --score
```

Standalone examples (no API key needed):

```bash
uv run examples/calendar_booking.py
uv run examples/showcase.py
uv run examples/date_extraction.py  # needs LITELLM-compatible API key
```

## Inspect AI integration

- Score individual skill steps with [Inspect AI](https://inspect.aisi.org.uk/) scorers.
- `skill_solver(skill)` wraps a skill as an Inspect `Solver`.
- When the Inspect model isn't `none/none`, `@lm` steps are rebound to call Inspect's model via `arun_skill`.
- `step_scorer(name, inner)` scores a named step value instead of the final completion.
- `step_check(name, predicate)` scores a named `StepResult`.
- Each model response appears as an assistant message in the Messages tab.
- Full request/response pairs with token usage show up in the Transcript as `ModelEvent` entries.

```python
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.scorer import match, model_graded_qa
from tk.llmbda.inspect import skill_solver, step_scorer

eval_task = Task(
    dataset=tickets,
    solver=skill_solver(support_triage),
    scorer=[
        step_scorer("λ::identifiers", match(location="any")),
        step_scorer("ψ::draft", model_graded_qa()),
        match(),
    ],
)

inspect_eval(eval_task, model="openai/gpt-4o-mini", log_dir="logs")
```

- `entry=` customises how `skill_solver` reads `TaskState` (default: `s.input_text`).
- `project=` stringifies non-`str` step values before the inner scorer sees them.
- Metrics default to the inner scorer's metrics; override with `metrics=[...]`.
- `model="none/none"` runs the skill with its native `@lm` callers (useful for scripted tests).
- `inspect_eval(...)` logs land under `./logs/`.

### Install and run

- Library: `pip install tk-llmbda[inspect]`
- Repo: `uv sync`
- Demo: `uv run examples/__main__.py <experiment> [--model <model>]`
- Scoring: `uv run examples/__main__.py <experiment> --score [--limit N]`
- Logs: `uv run inspect view`

## Development

```bash
# activate the pre-push hook (runs ruff + pytest before each push)
git config core.hooksPath .githooks

# skip the hook when you need to force-push a WIP
git push --no-verify
```
