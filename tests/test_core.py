import time

import pytest

from swarmagents.core import Swarm
from swarmagents.metrics import MetricsRegistry


def test_run_parallel_is_faster_than_serial() -> None:
    def slow_double(x: int) -> int:
        time.sleep(0.05)
        return x * 2

    swarm = Swarm(max_concurrency=8)
    start = time.perf_counter()
    results = swarm.run_sync([slow_double] * 8, range(8))
    elapsed = time.perf_counter() - start

    assert results == [x * 2 for x in range(8)]
    assert elapsed < 0.4


def test_concurrency_is_bounded() -> None:
    peak = 0
    active = 0

    def track(x: int) -> int:
        nonlocal peak, active
        active += 1
        peak = max(peak, active)
        time.sleep(0.02)
        active -= 1
        return x

    swarm = Swarm(max_concurrency=3)
    swarm.run_sync([track] * 12, range(12))
    assert peak <= 3


def test_metrics_recorded_on_success() -> None:
    registry = MetricsRegistry()

    async def ok(x: int) -> int:
        return x + 1

    swarm = Swarm(max_concurrency=4, registry=registry)
    results = swarm.run_sync([ok] * 3, range(3))

    assert results == [1, 2, 3]
    snap = registry.snapshot()
    assert snap["swarm.submitted"]["value"] == 3
    assert snap["swarm.success"]["value"] == 3
    assert snap["swarm.agent_duration_seconds"]["count"] == 3
    assert snap["swarm.total_duration_seconds"]["count"] == 1


def test_failures_are_counted_and_raised() -> None:
    registry = MetricsRegistry()

    def boom(x: int) -> int:
        if x == 1:
            raise ValueError("boom")
        return x

    swarm = Swarm(max_concurrency=2, registry=registry)
    with pytest.raises(ValueError, match="boom"):
        swarm.run_sync([boom] * 3, range(3))

    snap = registry.snapshot()
    assert snap["swarm.submitted"]["value"] == 3
    assert snap["swarm.success"]["value"] == 2
    assert snap["swarm.failure"]["value"] == 1


def test_timeout_is_enforced() -> None:
    registry = MetricsRegistry()

    def sleepy(x: int) -> int:
        time.sleep(0.2)
        return x

    swarm = Swarm(max_concurrency=1, timeout=0.05, registry=registry)
    with pytest.raises(TimeoutError):
        swarm.run_sync([sleepy], [1])
    assert registry.snapshot()["swarm.failure"]["value"] == 1


def test_invalid_concurrency_rejected() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        Swarm(max_concurrency=0)
