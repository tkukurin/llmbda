import asyncio

from tk.llmbda import Skill, SkillContext, StepResult, aiter_skill, iter_skill


def sync_stream_step(_ctx: SkillContext):
    yield "chunk 1 "
    yield "chunk 2"
    yield StepResult(value="done")


def test_sync_streaming_yields_chunks():
    skill = Skill("a", fn=sync_stream_step)
    items = list(iter_skill(skill, "entry"))

    assert items == [
        ("a", "chunk 1 "),
        ("a", "chunk 2"),
        ("a", StepResult(value="done")),
    ]


async def async_parallel_stream_step(_ctx: SkillContext):
    yield "p1"
    await asyncio.sleep(0.01)
    yield StepResult(value="p_done")


def test_async_parallel_streaming():
    skill = Skill(
        "root",
        parallel=True,
        steps=[Skill("p1", fn=async_parallel_stream_step)],
    )

    async def _run():
        return [item async for item in aiter_skill(skill, "entry")]

    items = asyncio.run(_run())
    assert items == [
        ("p1", "p1"),
        ("p1", StepResult(value="p_done")),
    ]
