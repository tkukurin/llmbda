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
    Skill,
    SkillContext,
    StepResult,
    check_skill,
    fst_match,
    iter_skill,
    last,
    lm,
    run_skill,
    strip_fences,
)

# %% [markdown]
# ## 1. Single deterministic step


# %%
def greet(ctx: SkillContext) -> str:
    """Greet by name."""
    return f"hello, {ctx.entry.get('name', 'world')}"


skill_greet = Skill(name="greeter", steps=[Skill("λ::greet", fn=greet)])
trace = run_skill(skill_greet, name="λ")
assert last(trace).value == "hello, λ"
assert "λ::greet" in trace
print(f"1. {last(trace).value}")

# %% [markdown]
# ## 2. Multi-step pipeline with ctx.prev and ctx.trace


# %%
def double(ctx: SkillContext) -> int:
    assert ctx.prev.value is None
    return ctx.entry["x"] * 2


def add_ten(ctx: SkillContext) -> int:
    return ctx.prev.value + 10


def format_result(ctx: SkillContext) -> str:
    return f"{ctx.trace['λ::double'].value} -> {ctx.prev.value}"


skill_math = Skill(
    name="math",
    steps=[
        Skill("λ::double", fn=double),
        Skill("λ::add_ten", fn=add_ten),
        Skill("λ::format", fn=format_result),
    ],
)
trace = run_skill(skill_math, x=5)
assert last(trace).value == "10 -> 20"
print(f"2. {last(trace).value}")

# %% [markdown]
# ## 3. Early-exit via orchestrator
#
# Steps fall through by default. An orchestrator stops on the first
# non-None result from its children.


# %%
def try_cache(ctx: SkillContext) -> StepResult:
    """Return cached result if available."""
    cache = {"known-key": "cached-value"}
    key = ctx.entry.get("key")
    if key in cache:
        return StepResult(value=cache[key])
    return StepResult()


def expensive_compute(_ctx: SkillContext) -> StepResult:
    return StepResult(value="computed-fresh")


skill_cache = Skill(
    name="cached",
    fn=fst_match,
    steps=[Skill("λ::cache", fn=try_cache), Skill("λ::compute", fn=expensive_compute)],
)

trace_hit = run_skill(Skill(name="s", steps=[skill_cache]), key="known-key")
assert last(trace_hit).value == "cached-value"

trace_miss = run_skill(Skill(name="s", steps=[skill_cache]), key="other")
assert last(trace_miss).value == "computed-fresh"
print(f"3. hit={last(trace_hit).value}, miss={last(trace_miss).value}")

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


skill_date = Skill(name="dates", steps=[Skill("ψ::extract", fn=extract_date)])
trace = run_skill(skill_date, text="let's meet on the 15th of January 2025")
assert last(trace).value == "2025-01-15"
print(f"4. {last(trace).value}")

# %% [markdown]
# ## 5. Mixed deterministic + LLM pipeline (regex-first, LLM fallback)


# %%
def extract_date_regex(ctx: SkillContext) -> StepResult:
    """Try regex before calling the LLM."""
    if m := _ISO_RE.search(ctx.entry["text"]):
        return StepResult(value=m.group(0), meta={"source": "regex"})
    return StepResult(meta={"reason": "no_match"})


skill_hybrid = Skill(
    name="hybrid_dates",
    fn=fst_match,
    steps=[
        Skill("λ::regex", fn=extract_date_regex),
        Skill("ψ::llm", fn=extract_date),
    ],
)

trace_regex = run_skill(
    Skill(name="s", steps=[skill_hybrid]), text="deadline is 2025-01-15"
)
assert last(trace_regex).value == "2025-01-15"
assert last(trace_regex).meta["source"] == "regex"

trace_llm = run_skill(
    Skill(name="s", steps=[skill_hybrid]),
    text="the fifteenth of January 2025",
)
assert last(trace_llm).value == "2025-01-15"
print(f"5. regex={last(trace_regex).value}, llm={last(trace_llm).value}")

# %% [markdown]
# ## 6. Nested composite skills


# %%
def normalize(ctx: SkillContext) -> str:
    return ctx.entry["text"].lower().strip()


def count_words(ctx: SkillContext) -> int:
    return len(ctx.trace["λ::normalize"].value.split())


def tag_length(ctx: SkillContext) -> str:
    n = ctx.trace["λ::count_words"].value
    return "short" if n < 5 else "medium" if n < 15 else "long"


skill_nested = Skill(
    name="analyzer",
    steps=[
        Skill(
            "preprocess",
            steps=[
                Skill("λ::normalize", fn=normalize),
                Skill("λ::count_words", fn=count_words),
            ],
        ),
        Skill(
            "classify",
            steps=[
                Skill("λ::tag_length", fn=tag_length),
            ],
        ),
    ],
)

trace = run_skill(skill_nested, text="  Hello World  ")
assert last(trace).value == "short"
assert trace["λ::normalize"].value == "hello world"
assert trace["λ::count_words"].value == 2
print(f"6. tag={last(trace).value}, words={trace['λ::count_words'].value}")

# %% [markdown]
# ## 7. Streaming execution with iter_skill

# %%
log: list[str] = []
for name, result in iter_skill(skill_math, x=3):
    log.append(f"{name}={result.value}")

assert log == ["λ::double=6", "λ::add_ten=16", "λ::format=6 -> 16"]
print(f"7. {log}")

# %% [markdown]
# ## 8. iter_skill with early break

# %%
seen = []
for name, _result in iter_skill(skill_nested, text="one two three four five six"):
    seen.append(name)
    if name == "λ::count_words":
        break

assert seen == ["λ::normalize", "λ::count_words"]
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
    value = ctx.trace["ψ::extract_date"].value
    valid = bool(value and _ISO_RE.fullmatch(str(value)))
    return StepResult(value=value, meta={"valid": valid})


def retry_extract_verify(ctx: SkillContext, steps: list[Skill]) -> StepResult:
    """Run extract→verify up to 3 times until valid."""
    inner = Skill(name="inner", steps=steps)
    for attempt in range(1, 4):
        r = run_skill(inner, ctx.entry)
        v = last(r)
        if v.meta.get("valid"):
            return StepResult(value=v.value, meta={"valid": True, "attempts": attempt})
    return StepResult(value=v.value, meta={"valid": False, "attempts": 3})


_call_count = 0
skill_retry = Skill(
    name="retry",
    fn=retry_extract_verify,
    steps=[
        Skill("ψ::extract_date", fn=extract_date_llm),
        Skill("λ::verify", fn=verify_date),
    ],
)
trace = run_skill(Skill(name="s", steps=[skill_retry]), text="next tuesday")
assert last(trace).value == "2025-06-01"
assert last(trace).meta["valid"] is True
assert last(trace).meta["attempts"] == 2
print(
    f"9. value={last(trace).value}, valid={last(trace).meta['valid']}, "
    f"attempts={last(trace).meta['attempts']}"
)

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
    intent = ctx.trace["ψ::classify"].value
    payload = json.dumps({"text": ctx.entry["text"], "intent": intent})
    raw = call(messages=[{"role": "user", "content": payload}])
    return StepResult(value=json.loads(raw))


skill_multi_model = Skill(
    name="booking",
    steps=[Skill("ψ::classify", fn=classify), Skill("ψ::confirm", fn=confirm)],
)
trace = run_skill(skill_multi_model, text="book Tuesday 3pm")
assert last(trace).value["confirmed"] is True
assert trace["ψ::classify"].value["intent"] == "booking"
intent = trace["ψ::classify"].value["intent"]
print(f"10. intent={intent}, slot={last(trace).value['slot']}")

# %% [markdown]
# ## 11. Test re-binding: swap model for testing via __wrapped__

# %%
captured_messages: list[list[dict[str, str]]] = []


def spy_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
    captured_messages.append(messages)
    return "spy-date"


rewrapped = lm(spy_model, system_prompt="test prompt")(extract_date.__wrapped__)
skill_spy = Skill(name="spy", steps=[Skill("ψ::extract", fn=rewrapped)])
trace = run_skill(skill_spy, {"text": "whenever"})
assert last(trace).value == "spy-date"
assert captured_messages[-1][0]["content"] == "test prompt"
print(f"11. rewrapped value={last(trace).value}")

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
        Skill("λ::real_step", fn=lambda _: StepResult(value=1)),
        Skill("λ::bad", fn=bad_step),
    ],
)
try:
    run_skill(skill_bad, {})
except KeyError as e:
    err = str(e)
    assert "typo" in err
    assert "λ::real_step" in err
    print(f"14. KeyError includes available steps: {err}")

# %% [markdown]
# ## 15. ctx.trace.get() for optional lookups


# %%
def optional_lookup(ctx: SkillContext) -> StepResult:
    maybe = ctx.trace.get("nonexistent")
    return StepResult(value="fallback" if maybe is None else maybe.value)


skill_get = Skill(name="opt", steps=[Skill("λ::a", fn=optional_lookup)])
trace = run_skill(skill_get, {})
assert last(trace).value == "fallback"
print(f"15. optional lookup={last(trace).value}")

# %% [markdown]
# ## 16. check_skill — static validation of trace references


# %%
def references_future(ctx: SkillContext) -> StepResult:
    return StepResult(value=ctx.trace["later"].value)


def later(_ctx: SkillContext) -> StepResult:
    return StepResult(value=1)


skill_forward_ref = Skill(
    name="fwd",
    steps=[Skill("λ::early", fn=references_future), Skill("λ::later", fn=later)],
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
        meta={
            "confidence": 0.92,
            "reason": "matched keyword 'refund'",
            "raw_model_output": '{"intent": "billing_refund"}',
        },
    )


skill_rich = Skill(name="rich", steps=[Skill("λ::classify", fn=rich_step)])
trace = run_skill(skill_rich, {})
assert last(trace).meta["confidence"] == 0.92
print(f"18. intent={last(trace).value}, confidence={last(trace).meta['confidence']}")

# %% [markdown]
# ## 19. Empty skill

# %%
trace = run_skill(Skill(name="noop"), {})
assert trace == {}
assert last(trace).value is None
print(f"19. empty skill: trace={trace}")

# %% [markdown]
# ## 20. Full trace access

# %%
trace = run_skill(skill_nested, text="a b c d e f g h i j k l m n o p")
assert set(trace) == {"λ::normalize", "λ::count_words", "λ::tag_length"}
assert last(trace).value == "long"
print(f"20. trace keys: {list(trace)}, final={last(trace).value}")

# %%
print("\nAll 20 use cases passed.")
