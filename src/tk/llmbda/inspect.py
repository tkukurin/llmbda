"""Adapters for running `tk.llmbda` skills inside Inspect AI."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import copy
from functools import wraps
from typing import TYPE_CHECKING, Any

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.scorer import Score, accuracy, mean, scorer, stderr
from inspect_ai.solver import solver

from tk.llmbda import Skill, StepResult, arun_skill, last, run_skill

if TYPE_CHECKING:
    from collections.abc import Callable

    from inspect_ai.scorer import Scorer
    from inspect_ai.solver import Solver, TaskState

DEFAULT_TRACE_KEY = "llmbda.trace"


def _get_model(name: str):
    """Lazy wrapper around inspect_ai.model.get_model (patchable in tests)."""
    from inspect_ai.model import get_model  # noqa: PLC0415

    return get_model(name)


def _to_chat_messages(messages: list[dict[str, str]]) -> list:
    """Convert plain dicts to Inspect ChatMessage objects."""
    _cls = {
        "system": ChatMessageSystem,
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
    }
    return [
        _cls.get(m["role"], ChatMessageUser)(content=m["content"]) for m in messages
    ]


def _make_async_caller(model_name: str) -> tuple[Callable[..., Any], list]:
    """Create an async caller and capture request/response pairs."""
    _cache: list = []
    message_log: list[tuple[list, str]] = []

    async def async_caller(*, messages: list[dict[str, str]], **_kw: Any) -> str:
        if not _cache:
            _cache.append(_get_model(model_name))
        chat_msgs = _to_chat_messages(messages)
        result = await _cache[0].generate(chat_msgs)
        message_log.append((chat_msgs, result.completion))
        return result.completion

    return async_caller, message_log


def _rebind_skill_async(skill: Skill, async_caller: Callable[..., Any]) -> Skill:
    """Copy a skill tree and rebind `@lm` steps to an async caller."""
    if skill.fn and hasattr(skill.fn, "lm_system_prompt"):
        sys_prompt = skill.fn.lm_system_prompt
        original = skill.fn.__wrapped__

        if sys_prompt:

            async def bound(*, messages: list[dict[str, str]], **kwargs: Any) -> Any:
                return await async_caller(
                    messages=[{"role": "system", "content": sys_prompt}, *messages],
                    **kwargs,
                )
        else:
            bound = async_caller

        if asyncio.iscoroutinefunction(original):

            @wraps(original)
            async def new_fn(ctx, *args):
                return await original(ctx, *args, bound)
        else:

            @wraps(original)
            async def new_fn(ctx, *args):
                loop = asyncio.get_running_loop()
                parent_ctx = contextvars.copy_context()

                def sync_call(*, messages: list[dict[str, str]], **kw: Any) -> str:
                    coro = bound(messages=messages, **kw)
                    return _await_in_context(loop, coro, parent_ctx)

                return await loop.run_in_executor(None, original, ctx, *args, sync_call)

        new_fn.lm_system_prompt = sys_prompt  # type: ignore[attr-defined]
        new_fn.lm_model = skill.fn.lm_model  # type: ignore[attr-defined]
        new_fn_final: Any = new_fn
    else:
        new_fn_final = skill.fn
    new_steps = [_rebind_skill_async(s, async_caller) for s in skill.steps]
    return Skill(
        name=skill.name, fn=new_fn_final, steps=new_steps, description=skill.description
    )


def _await_in_context(
    loop: asyncio.AbstractEventLoop,
    coro: Any,
    context: contextvars.Context,
) -> Any:
    """Run `coro` on `loop` inside `context` and block the caller thread."""
    future: concurrent.futures.Future = concurrent.futures.Future()

    async def _run():
        try:
            result = await coro
            future.set_result(result)
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)

    loop.call_soon_threadsafe(loop.create_task, _run(), context=context)
    return future.result()


def skill_solver(
    skill: Skill,
    *,
    entry: Callable[[TaskState], Any] | None = None,
    trace_key: str = DEFAULT_TRACE_KEY,
) -> Solver:
    """Run *skill* once per sample; route @lm calls through Inspect's model.

    Model calls appear in Inspect's transcript with full request/response pairs.
    """
    extract = entry or (lambda s: s.input_text)

    @solver
    def _factory():
        async def solve(state, _generate):
            model_name = str(state.model)
            use_inspect_model = model_name != "none/none"

            if use_inspect_model:
                async_caller, message_log = _make_async_caller(model_name)
                patched = _rebind_skill_async(skill, async_caller)
                trace = await arun_skill(patched, extract(state))
                for msgs, completion in message_log:
                    state.messages.extend(msgs)
                    state.messages.append(ChatMessageAssistant(content=completion))
            else:
                trace = run_skill(skill, extract(state))
            final = last(trace)
            state.output = ModelOutput.from_content(
                model=model_name,
                content="" if final.value is None else str(final.value),
            )
            if state.metadata is None:
                state.metadata = {}
            state.metadata[trace_key] = trace
            return state

        return solve

    return _factory()


def step_scorer(
    step_name: str,
    inner: Scorer,
    *,
    trace_key: str = DEFAULT_TRACE_KEY,
    project: Callable[[Any], str] = str,
    metrics: list | None = None,
) -> Scorer:
    """Apply *inner* scorer to the named step's value instead of final completion."""
    resolved_metrics = metrics or _inherit_metrics(inner) or [accuracy(), stderr()]

    @scorer(metrics=resolved_metrics, name=f"step[{step_name}]")
    def _factory():
        async def score(state, target):
            trace = (state.metadata or {}).get(trace_key) or {}
            if step_name not in trace:
                available = ", ".join(trace) or "(none)"
                raise KeyError(f"{step_name!r} not in trace (available: {available})")
            step = trace[step_name]
            shadow = copy.copy(state)
            shadow.output = ModelOutput.from_content(
                model=str(state.model),
                content=project(step.value),
            )
            return await inner(shadow, target)

        return score

    return _factory()


def step_check(
    step_name: str,
    predicate: Callable[[StepResult], float | bool | Score],
    *,
    trace_key: str = DEFAULT_TRACE_KEY,
    metrics: list | None = None,
) -> Scorer:
    """Score a step by predicate on its StepResult (value + meta)."""
    resolved_metrics = metrics or [mean(), stderr()]

    @scorer(metrics=resolved_metrics, name=f"check[{step_name}]")
    def _factory():
        async def score(state, target):  # noqa: ARG001
            trace = (state.metadata or {}).get(trace_key) or {}
            if step_name not in trace:
                available = ", ".join(trace) or "(none)"
                raise KeyError(f"{step_name!r} not in trace (available: {available})")
            out = predicate(trace[step_name])
            if isinstance(out, Score):
                return out
            val = float(out) if isinstance(out, (int, float)) else (1.0 if out else 0.0)
            return Score(value=val)

        return score

    return _factory()


def _inherit_metrics(inner: Scorer) -> list | None:
    """Best-effort read of metrics baked into *inner* at definition time."""
    try:
        from inspect_ai.scorer._scorer import scorer_metrics  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return list(scorer_metrics(inner)) or None
    except (AttributeError, KeyError, TypeError):
        return None


_PASSTHROUGH_REGISTRY: dict[str, Any] = {}


_DEFAULT_CONFIG = GenerateConfig()


@modelapi(name="llmbda")
class _PassthroughModelAPI(ModelAPI):
    """Inspect `ModelAPI` that delegates to a registered caller."""

    def __init__(
        self,
        model_name: str = "llmbda/default",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig = _DEFAULT_CONFIG,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars or [],
            config=config,
        )

    async def generate(
        self,
        input,  # noqa: A002
        tools,  # noqa: ARG002
        tool_choice,  # noqa: ARG002
        config,  # noqa: ARG002
    ):
        name = (
            self.model_name.split("/", 1)[-1]
            if "/" in self.model_name
            else self.model_name
        )
        fn = _PASSTHROUGH_REGISTRY[name]
        messages = []
        for m in input:
            if isinstance(m, ChatMessageSystem):
                messages.append({"role": "system", "content": m.content})
            elif isinstance(m, ChatMessageUser):
                messages.append({"role": "user", "content": m.content})
            else:
                messages.append({"role": "assistant", "content": m.content})
        result = fn(messages=messages)
        if asyncio.iscoroutine(result):
            result = await result
        return ModelOutput.from_content(model=self.model_name, content=result)


def passthrough_model(fn: Any, name: str = "default") -> str:
    """Register a callable as an Inspect model and return its model name."""
    _PASSTHROUGH_REGISTRY[name] = fn
    return f"llmbda/{name}"


__all__ = [
    "DEFAULT_TRACE_KEY",
    "passthrough_model",
    "skill_solver",
    "step_check",
    "step_scorer",
]
