# %% [markdown]
# # Compile sketch
#
# Walk a `Skill` and produce a single `skill.md` from step metadata.
#
# - **Leaf (`fn`, no `steps`)** — deterministic or LLM.
#   - Deterministic → source code extracted via `inspect.getsource`,
#     included as a fenced code block.
#   - LLM (`lm_system_prompt` present) → natural-language instructions
#     assembled from `description` and `lm_system_prompt`.
# - **Composite (no `fn`, has `steps`)** → section heading, recurse.
# - **Orchestrator (`fn` + `steps`)** → render the fn, then recurse
#   into declared children for introspection.
# - Each renderer is a standalone function; `compile_skill` assembles
#   them into a single markdown string.

# %%
from __future__ import annotations

import inspect
import textwrap
from typing import Any

from tk.llmbda import Skill


def _is_lm(skill: Skill) -> bool:
    return skill.fn is not None and bool(getattr(skill.fn, "lm_system_prompt", None))


def _is_orchestrator(skill: Skill) -> bool:
    return skill.fn is not None and bool(skill.steps)


def _is_leaf(skill: Skill) -> bool:
    return skill.fn is not None and not skill.steps


def _source_for(fn: Any) -> str:
    """Best-effort source extraction; falls back to '(source unavailable)'."""
    target = getattr(fn, "__wrapped__", fn)
    try:
        return textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):
        return "(source unavailable)\n"


def render_fn_leaf(skill: Skill, heading: str = "###") -> str:
    """Render a deterministic leaf as a code-reference block."""
    lines = [f"{heading} {skill.name}", "", "**Type:** deterministic function"]
    if skill.description:
        lines += [f"**Description:** {skill.description}", ""]
    else:
        lines.append("")
    lines += ["```python", _source_for(skill.fn).rstrip(), "```", ""]
    return "\n".join(lines)


def render_lm_leaf(skill: Skill, heading: str = "###") -> str:
    """Render an LLM leaf as natural-language instructions."""
    prompt = getattr(skill.fn, "lm_system_prompt", "") or ""
    lines = [f"{heading} {skill.name}", "", "**Type:** LLM"]
    if skill.description:
        lines += [f"**Description:** {skill.description}", ""]
    if prompt:
        lines += ["**System prompt:**", "", f"> {prompt}", ""]
    return "\n".join(lines)


def render_orchestrator(skill: Skill, heading: str = "###") -> str:
    """Render an orchestrator: its own fn, then declared children."""
    tag = "LLM orchestrator" if _is_lm(skill) else "orchestrator"
    lines = [f"{heading} {skill.name}", "", f"**Type:** {tag}"]
    if skill.description:
        lines += [f"**Description:** {skill.description}", ""]
    else:
        lines.append("")
    prompt = getattr(skill.fn, "lm_system_prompt", "") or ""
    if prompt:
        lines += ["**System prompt:**", "", f"> {prompt}", ""]
    lines += ["```python", _source_for(skill.fn).rstrip(), "```", ""]
    if skill.steps:
        lines.append(f"**Children ({len(skill.steps)}):**\n")
        sub = heading + "#"
        for child in skill.steps:
            lines.append(render_skill_node(child, heading=sub))
    return "\n".join(lines)


def render_composite(skill: Skill, heading: str = "###") -> str:
    """Render a composite (no fn, just groups children)."""
    lines = [f"{heading} {skill.name}", ""]
    sub = heading + "#"
    for child in skill.steps:
        lines.append(render_skill_node(child, heading=sub))
    return "\n".join(lines)


def render_skill_node(skill: Skill, heading: str = "###") -> str:
    if _is_orchestrator(skill):
        return render_orchestrator(skill, heading=heading)
    if _is_lm(skill):
        return render_lm_leaf(skill, heading=heading)
    if _is_leaf(skill):
        return render_fn_leaf(skill, heading=heading)
    return render_composite(skill, heading=heading)


def compile_skill(skill: Skill) -> str:
    """Assemble a full skill.md from a Skill tree."""
    lines = [f"# {skill.name}", ""]
    if _is_leaf(skill) or _is_orchestrator(skill):
        lines.append(render_skill_node(skill, heading="##"))
    else:
        for child in skill.steps:
            lines.append(render_skill_node(child, heading="##"))
    return "\n".join(lines).rstrip() + "\n"


# %% [markdown]
# ## Example 1 — calendar booking (orchestrator + regex leaves)

# %%
import json
import re

from tk.llmbda import LMCaller, SkillContext, StepResult, lm, run_skill, strip_fences

WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
)


def parse_weekday(ctx: SkillContext) -> StepResult:
    """Find an explicit weekday name (Monday..Sunday)."""
    text = ctx.entry["text"].lower()
    for day in WEEKDAYS:
        if re.search(rf"\b{day}\b", text):
            return StepResult(day.capitalize(), {"reason": "matched"})
    return StepResult(None, {"reason": "no_weekday"})


def parse_time(ctx: SkillContext) -> StepResult:
    """Find a clock time like '3pm', '15:00', or a range '9-10am'."""
    return StepResult(None, {"reason": "no_time"})


def parse_duration(ctx: SkillContext) -> StepResult:
    """Find a duration phrase like '30 minutes' or '2 hrs'."""
    return StepResult(None, {"reason": "no_duration"})


def parse_topic(ctx: SkillContext) -> StepResult:
    """Find a topic phrase introduced by 'about' or 're:'."""
    return StepResult(None, {"reason": "no_topic"})


def fake(*, messages: list[dict[str, str]], **_kw: Any) -> str:
    return "{}"


VERIFY_PROMPT = (
    "You are a calendar booking verifier. Cross-check the prior "
    "findings against the text: confirm, correct, fill gaps, "
    "flag ambiguity."
)


@lm(fake, system_prompt=VERIFY_PROMPT)
def verify(ctx: SkillContext, steps: list[Skill], call: LMCaller) -> StepResult:
    """Run parser children, cross-check extractions against the raw text."""
    inner = Skill(name="_parse", steps=steps)
    r = run_skill(inner, ctx.entry)
    payload = json.dumps({
        "text": ctx.entry["text"],
        "prior_steps": [
            {"name": s.name, "description": s.description, "value": r.trace[s.name].value}
            for s in steps if s.name in r.trace
        ],
    })
    raw = call(messages=[{"role": "user", "content": payload}])
    return StepResult(json.loads(strip_fences(raw)))


book_meeting = Skill(
    name="book_meeting",
    fn=verify,
    steps=[
        Skill("λ::weekday", fn=parse_weekday),
        Skill("λ::time", fn=parse_time),
        Skill("λ::duration", fn=parse_duration),
        Skill("λ::topic", fn=parse_topic),
    ],
)

print(compile_skill(book_meeting))
print("=" * 60)


# %% [markdown]
# ## Example 2 — retry orchestrator with LLM extract + deterministic validate

# %%
_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


@lm(fake, system_prompt="Extract a date from the text. Return ONLY ISO-8601.")
def extract_date_lm(ctx: SkillContext, call: Any) -> StepResult:
    """Extract a date via LLM."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(raw.strip(), {"source": "lm"})


def validate_iso(ctx: SkillContext) -> StepResult:
    """Check whether the latest extraction is a valid ISO-8601 date."""
    prev = ctx.trace.get("ψ::extract")
    if prev and _ISO_RE.fullmatch(str(prev.value or "")):
        return StepResult(prev.value, {"valid": True})
    return StepResult(None, {"valid": False})


def retry_extract_verify(ctx: SkillContext, steps: list[Skill]) -> StepResult:
    """Run extract→validate up to 3 times until valid."""
    inner = Skill(name="inner", steps=steps)
    for attempt in range(1, 4):
        r = run_skill(inner, ctx.entry)
        if r.metadata.get("valid"):
            return StepResult(
                value=r.value,
                metadata={"valid": True, "attempts": attempt},
                resolved_by=r.resolved_by,
            )
    return StepResult(
        value=r.value,
        metadata={"valid": False, "attempts": 3},
        resolved_by=r.resolved_by,
    )


dates_skill = Skill(
    name="dates",
    fn=retry_extract_verify,
    steps=[
        Skill("ψ::extract", fn=extract_date_lm),
        Skill("λ::validate", fn=validate_iso),
    ],
)

print(compile_skill(dates_skill))