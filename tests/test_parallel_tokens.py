import asyncio
import time

from pydanticswarm import AgentSpec, Subtask
from pydanticswarm.merge import MergedAgent, MergedSwarm
from pydanticswarm.parallel import WorkerPool, bounded_gather
from pydanticswarm.tokens import PromptSpec, TokenMeter, estimate_tokens


class FakeBackend:
    async def generate(self, prompt: str) -> str:
        await asyncio.sleep(0.01)
        return "ok"


# ---------- parallelism ----------


def test_pool_limits_concurrency():
    async def flow():
        concurrent = 0
        peak = 0

        async def job():
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1
            return "done"

        pool = WorkerPool(max_workers=4)
        for i in range(20):
            await pool.submit(i, job)
        results, errors = await pool.run()
        assert len(results) == 20 and not errors
        assert pool.stats.max_concurrency_observed <= 4
        assert peak <= 4
        assert pool.stats.completed == 20

    asyncio.run(flow())


def test_pool_backpressure_bounds_queue():
    async def flow():
        pool = WorkerPool(max_workers=2, queue_max=4)

        async def job():
            await asyncio.sleep(0.02)
            return 1

        async def producer():
            for i in range(10):
                await pool.submit(i, job)

        # producer runs concurrently with workers: queue never exceeds max
        await asyncio.gather(producer(), pool.run())
        assert pool.stats.queue_high_water <= 4
        assert pool.stats.completed == 10

    asyncio.run(flow())


def test_pool_isolates_failures():
    async def flow():
        async def boom():
            raise RuntimeError("dead")

        async def ok():
            return "ok"

        pool = WorkerPool(max_workers=2)
        await pool.submit(0, boom)
        await pool.submit(1, ok)
        results, errors = await pool.run()
        assert 0 in errors and "RuntimeError" in errors[0]
        assert results[1] == "ok"
        assert pool.stats.failed == 1

    asyncio.run(flow())


def test_bounded_gather_caps_parallel():
    async def flow():
        running = 0
        peak = 0

        async def job():
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            return "x"

        results, _ = await bounded_gather([job for _ in range(15)], max_workers=3)
        assert len(results) == 15
        assert peak <= 3

    asyncio.run(flow())


def test_pool_rejects_bad_worker_count():
    try:
        WorkerPool(max_workers=0)
        raise AssertionError("should raise")
    except ValueError as e:
        assert "max_workers" in str(e)


# ---------- tokens ----------


def test_prompt_prefix_is_stable_and_cacheable():
    a = PromptSpec(agent_name="coder", agent_skills=["code"], task_description="task A")
    b = PromptSpec(agent_name="coder", agent_skills=["code"], task_description="task B")
    # Prefix identical -> provider prompt-cache prefix hit
    i = next(
        i
        for i in range(min(len(a.render()), len(b.render())))
        if a.render()[i] != b.render()[i]
    )
    prefix_a = a.render()[:i]
    prefix_b = b.render()[:i]
    assert prefix_a == prefix_b
    assert "task A" in a.render() and "task B" in b.render()


def test_token_meter_tracks_and_budgets():
    meter = TokenMeter(budget=100)
    n = meter.record("agent", "a" * 400, "b" * 200, cached_prefix=True)
    assert n > 0
    u = meter.usage_for("agent")
    assert u.calls == 1 and u.cache_hits == 1
    assert meter.total > 0
    # Budget exceeded -> returns 0, caller skips
    over = meter.record("agent", "a" * 4000, "b" * 2000)
    assert over == 0
    s = meter.summary()
    assert s["calls"] == 1


def test_estimate_tokens_sane():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 100) == 25


def test_merged_swarm_tracks_tokens():
    async def flow():
        swarm = MergedSwarm(
            agents=[MergedAgent(AgentSpec(name="a", skills=["x"]), FakeBackend())],
            token_budget=10_000,
        )
        await swarm.run(
            "goal",
            subtasks=[
                Subtask(id="1", description="do"),
                Subtask(id="2", description="it"),
            ],
        )
        s = swarm.tokens.summary()
        assert s["calls"] == 2
        assert s["cache_hits"] == 2
        assert s["total_tokens"] > 0

    asyncio.run(flow())


def test_merged_swarm_respects_budget():
    async def flow():
        swarm = MergedSwarm(
            agents=[MergedAgent(AgentSpec(name="a", skills=["x"]), FakeBackend())],
            token_budget=1,
        )
        await swarm.run(
            "goal",
            subtasks=[Subtask(id="1", description="do")],
        )
        assert swarm.tokens.total == 0  # nothing recorded: budget too small

    asyncio.run(flow())


def test_pool_is_faster_than_serial():
    async def flow():
        start = time.monotonic()
        pool = WorkerPool(max_workers=4)
        for i in range(8):
            await pool.submit(i, FakeBackend().generate)
        await pool.run()
        parallel_s = time.monotonic() - start
        # 8 x 10ms serial = 80ms; with 4 workers ≈ 20ms + overhead
        assert parallel_s < 0.08

    asyncio.run(flow())
