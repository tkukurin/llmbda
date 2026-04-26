# %% [markdown]
# # tk.llmbda showcase
#
# Every common use case in one runnable file. No external dependencies.

# %%
from __future__ import annotations

import json
import re
from typing import Any

from tk.llmbda import (
    ROOT,
    Skill,
    SkillContext,
    StepResult,
    check_skill,
    iter_skill,
    lm,
    run_skill,
    strip_fences,
)

# %% [markdown]
# ## 1. Single deterministic step

# %%
def greet(ctx: SkillContext) -> StepResult:
    """Greet by name."""
    name = ctx.entry.get("name", "world")
    return StepResult(value=f"hello, {name}")

skill_greet = Skill(name="greeter", steps=[Skill("greet", fn=greet)])
r = run_skill(skill_greet, {"name": "λ"})
assert r.value == "hello, λ"
assert r.resolved_by == "greet"
print(f"1. {r.value}")

# %% [markdown]
# ## 2. Multi-step pipeline with ctx.prev and ctx.trace

# %%
def double(ctx: SkillContext) -> StepResult:
    assert ctx.prev is ROOT
    return StepResult(value=ctx.entry["x"] * 2)

def add_ten(ctx: SkillContext) -> StepResult:
    return StepResult(value=ctx.prev.value + 10)

def format_result(ctx: SkillContext) -> StepResult:
    doubled = ctx.trace["double"].value
    added = ctx.prev.value
    return StepResult(value=f"{doubled} -> {added}")

skill_math = Skill(
    name="math",
    steps=[
        Skill("double", fn=double),
        Skill("add_ten", fn=add_ten),
        Skill("format", fn=format_result),
    ],
)
r = run_skill(skill_math, {"x": 5})
assert r.value == "10 -> 20"
print(f"2. {r.value}")

# %% [markdown]
# ## 3. Short-circuit with resolved=True
#
# Steps fall through by default (`resolved=False`). Setting `resolved=True`
# skips remaining steps.

# %%
def try_cache(ctx: SkillContext) -> StepResult:
    """Return cached result if available."""
    cache = {"known-key": "cached-value"}
    key = ctx.entry.get("key")
    if key in cache:
        return StepResult(value=cache[key], resolved=True)
    return StepResult(value=None, metadata={"reason": "cache_miss"})

def expensive_compute(_ctx: SkillContext) -> StepResult:
    return StepResult(value="computed-fresh")

skill_cache = Skill(
    name="cached",
    steps=[Skill("cache", fn=try_cache), Skill("compute", fn=expensive_compute)],
)

r_hit = run_skill(skill_cache, {"key": "known-key"})
assert r_hit.resolved_by == "cache"
assert r_hit.value == "cached-value"

r_miss = run_skill(skill_cache, {"key": "other"})
assert r_miss.resolved_by == "compute"
assert r_miss.value == "computed-fresh"
print(f"3. hit={r_hit.resolved_by}, miss={r_miss.resolved_by}")

# %% [markdown]
# ## 4. LLM step with @lm decorator and fake model

# %%
_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

def fake_date_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:  # noqa: ARG001
    return "2025-01-15"

@lm(fake_date_model, system_prompt="Extract a date. Return ONLY ISO-8601.")
def extract_date(ctx: SkillContext, call) -> StepResult:
    """Extract a date from natural language."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(value=raw.strip())

skill_date = Skill(name="dates", steps=[Skill("extract", fn=extract_date)])
r = run_skill(skill_date, {"text": "let's meet on the 15th of January 2025"})
assert r.value == "2025-01-15"
print(f"4. {r.value}")

# %% [markdown]
# ## 5. Mixed deterministic + LLM pipeline (regex-first, LLM fallback)

# %%
def extract_date_regex(ctx: SkillContext) -> StepResult:
    """Try regex before calling the LLM."""
    if m := _ISO_RE.search(ctx.entry["text"]):
        return StepResult(value=m.group(0), metadata={"source": "regex"}, resolved=True)
    return StepResult(value=None, metadata={"reason": "no_match"})

skill_hybrid = Skill(
    name="hybrid_dates",
    steps=[
        Skill("λ::regex", fn=extract_date_regex),
        Skill("ψ::llm", fn=extract_date),
    ],
)

r_regex = run_skill(skill_hybrid, {"text": "deadline is 2025-01-15"})
assert r_regex.resolved_by == "λ::regex"
assert r_regex.metadata["source"] == "regex"

r_llm = run_skill(skill_hybrid, {"text": "the fifteenth of January 2025"})
assert r_llm.resolved_by == "ψ::llm"
print(f"5. regex={r_regex.resolved_by}, llm={r_llm.resolved_by}")

# %% [markdown]
# ## 6. Nested composite skills

# %%
def normalize(ctx: SkillContext) -> StepResult:
    return StepResult(value=ctx.entry["text"].lower().strip())

def count_words(ctx: SkillContext) -> StepResult:
    return StepResult(value=len(ctx.trace["normalize"].value.split()))

def tag_length(ctx: SkillContext) -> StepResult:
    n = ctx.trace["count_words"].value
    tag = "short" if n < 5 else "medium" if n < 15 else "long"
    return StepResult(value=tag)

skill_nested = Skill(
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

r = run_skill(skill_nested, {"text": "  Hello World  "})
assert r.value == "short"
assert r.trace["normalize"].value == "hello world"
assert r.trace["count_words"].value == 2
print(f"6. tag={r.value}, words={r.trace['count_words'].value}")

# %% [markdown]
# ## 7. Streaming execution with iter_skill

# %%
log: list[str] = []
for name, result in iter_skill(skill_math, {"x": 3}):
    log.append(f"{name}={result.value}")

assert log == ["double=6", "add_ten=16", "format=6 -> 16"]
print(f"7. {log}")

# %% [markdown]
# ## 8. iter_skill with early break

# %%
seen = []
for name, _result in iter_skill(skill_nested, {"text": "one two three four five six"}):
    seen.append(name)
    if name == "count_words":
        break

assert seen == ["normalize", "count_words"]
print(f"8. stopped after: {seen}")

# %% [markdown]
# ## 9. Parent retry skill orchestrates extract + verify
#
# A `Skill` can have both `fn` and `steps`. The children are declared
# on the skill for introspection and static checks; the fn orchestrates
# them. Here `retry` declares extract + verify as children and loops
# them until the joint result is valid.

# %%
_call_count = 0

def fake_flaky_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:  # noqa: ARG001
    """Returns invalid output on first call, valid on second."""
    global _call_count  # noqa: PLW0603
    _call_count += 1
    return "not-a-date" if _call_count == 1 else "2025-06-01"

@lm(fake_flaky_model, system_prompt="Extract date as YYYY-MM-DD.")
def extract_date_llm(ctx: SkillContext, call) -> StepResult:
    """Extract a date via LLM."""
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(value=raw.strip())

def verify_date(ctx: SkillContext) -> StepResult:
    """Check whether the extracted value is a valid ISO date."""
    value = ctx.trace["ψ::extract"].value
    valid = bool(value and _ISO_RE.fullmatch(str(value)))
    return StepResult(value=value, metadata={"valid": valid})

def retry_extract_verify(ctx: SkillContext) -> StepResult:
    """Run extract→verify up to 3 times until valid."""
    inner = Skill(name="inner", steps=ctx.skills)
    for attempt in range(1, 4):
        r = run_skill(inner, ctx.entry)
        if r.metadata.get("valid"):
            return StepResult(
                value=r.value,
                metadata={"valid": True, "attempts": attempt},
            )
    return StepResult(
        value=r.value,
        metadata={"valid": False, "attempts": 3},
    )

_call_count = 0
skill_retry = Skill(
    name="retry",
    fn=retry_extract_verify,
    steps=[
        Skill("ψ::extract", fn=extract_date_llm),
        Skill("ψ::verify", fn=verify_date),
    ],
)
r = run_skill(skill_retry, {"text": "next tuesday"})
assert r.value == "2025-06-01"
assert r.metadata["valid"] is True
assert r.metadata["attempts"] == 2
print(f"9. value={r.value}, valid={r.metadata['valid']}, "
      f"attempts={r.metadata['attempts']}")

# %% [markdown]
# ## 10. Per-step model binding (different models in one skill)

# %%
def model_cheap(*, messages: list[dict[str, str]], **_kw: Any) -> str:  # noqa: ARG001
    return json.dumps({"intent": "booking"})

def model_expensive(*, messages: list[dict[str, str]], **_kw: Any) -> str:  # noqa: ARG001
    return json.dumps({"confirmed": True, "slot": "Tuesday 3pm"})

@lm(model_cheap, system_prompt="Classify intent.")
def classify(ctx: SkillContext, call) -> StepResult:
    raw = call(messages=[{"role": "user", "content": ctx.entry["text"]}])
    return StepResult(value=json.loads(raw))

@lm(model_expensive, system_prompt="Confirm booking.")
def confirm(ctx: SkillContext, call) -> StepResult:
    intent = ctx.trace["classify"].value
    payload = json.dumps({"text": ctx.entry["text"], "intent": intent})
    raw = call(messages=[{"role": "user", "content": payload}])
    return StepResult(value=json.loads(raw))

skill_multi_model = Skill(
    name="booking",
    steps=[Skill("classify", fn=classify), Skill("confirm", fn=confirm)],
)
r = run_skill(skill_multi_model, {"text": "book Tuesday 3pm"})
assert r.value["confirmed"] is True
assert r.trace["classify"].value["intent"] == "booking"
print(f"10. intent={r.trace['classify'].value['intent']}, slot={r.value['slot']}")

# %% [markdown]
# ## 11. Test re-binding: swap model for testing via __wrapped__

# %%
captured_messages: list[list[dict[str, str]]] = []

def spy_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
    captured_messages.append(messages)
    return "spy-date"

rewrapped = lm(spy_model, system_prompt="test prompt")(extract_date.__wrapped__)
skill_spy = Skill(name="spy", steps=[Skill("extract", fn=rewrapped)])
r = run_skill(skill_spy, {"text": "whenever"})
assert r.value == "spy-date"
assert captured_messages[-1][0]["content"] == "test prompt"
print(f"11. rewrapped value={r.value}")

# %% [markdown]
# ## 12. Introspecting @lm attributes

# %%
assert extract_date.lm_system_prompt == "Extract a date. Return ONLY ISO-8601."
assert extract_date.lm_model is fake_date_model
print(f"12. prompt={extract_date.lm_system_prompt!r}")

# %% [markdown]
# ## 13. Skill.description from docstring vs explicit

# %%
s_auto = Skill("auto", fn=extract_date)
assert s_auto.description == "Extract a date from natural language."

s_explicit = Skill("explicit", fn=extract_date, description="Custom override")
assert s_explicit.description == "Custom override"
print(f"13. auto={s_auto.description!r}, explicit={s_explicit.description!r}")

# %% [markdown]
# ## 14. _Trace KeyError with available-step diagnostics

# %%
def bad_step(ctx: SkillContext) -> StepResult:
    return StepResult(value=ctx.trace["typo"])

skill_bad = Skill(
    name="bad",
    steps=[
        Skill("real_step", fn=lambda _: StepResult(value=1)),
        Skill("bad", fn=bad_step),
    ],
)
try:
    run_skill(skill_bad, {})
except KeyError as e:
    err = str(e)
    assert "typo" in err
    assert "real_step" in err
    print(f"14. KeyError includes available steps: {err}")

# %% [markdown]
# ## 15. ctx.trace.get() for optional lookups

# %%
def optional_lookup(ctx: SkillContext) -> StepResult:
    maybe = ctx.trace.get("nonexistent")
    return StepResult(value="fallback" if maybe is None else maybe.value)

skill_get = Skill(name="opt", steps=[Skill("a", fn=optional_lookup)])
r = run_skill(skill_get, {})
assert r.value == "fallback"
print(f"15. optional lookup={r.value}")

# %% [markdown]
# ## 16. check_skill — static validation of trace references

# %%
def references_future(ctx: SkillContext) -> StepResult:
    return StepResult(value=ctx.trace["later"].value)

def later(_ctx: SkillContext) -> StepResult:
    return StepResult(value=1)

skill_forward_ref = Skill(
    name="fwd",
    steps=[Skill("early", fn=references_future), Skill("later", fn=later)],
)
issues = check_skill(skill_forward_ref)
assert any("later" in i for i in issues)
print(f"16. static check caught: {issues}")

clean_issues = check_skill(skill_math)
assert clean_issues == []
print(f"    clean skill: {clean_issues}")

# %% [markdown]
# ## 17. strip_fences utility

# %%
fenced = '```json\n{"a": 1}\n```'
assert strip_fences(fenced) == '{"a": 1}'
assert strip_fences("plain text") == "plain text"
print(f"17. strip_fences: {strip_fences(fenced)!r}")

# %% [markdown]
# ## 18. Metadata for auxiliary context (confidence, reasons, raw output)

# %%
def rich_step(_ctx: SkillContext) -> StepResult:
    return StepResult(
        value="billing_refund",
        metadata={
            "confidence": 0.92,
            "reason": "matched keyword 'refund'",
            "raw_model_output": '{"intent": "billing_refund"}',
        },
    )

skill_rich = Skill(name="rich", steps=[Skill("classify", fn=rich_step)])
r = run_skill(skill_rich, {})
assert r.metadata["confidence"] == 0.92
print(f"18. intent={r.value}, confidence={r.metadata['confidence']}")

# %% [markdown]
# ## 19. Empty skill

# %%
r = run_skill(Skill(name="noop"), {})
assert r.value is None
assert r.resolved_by == "(empty)"
print(f"19. empty skill: resolved_by={r.resolved_by}")

# %% [markdown]
# ## 20. Full trace in SkillResult

# %%
r = run_skill(skill_nested, {"text": "a b c d e f g h i j k l m n o p"})
assert set(r.trace) == {"normalize", "count_words", "tag_length"}
assert r.value == "long"
print(f"20. trace keys: {list(r.trace)}, final={r.value}")

# %%
print("\nAll 20 use cases passed.")
