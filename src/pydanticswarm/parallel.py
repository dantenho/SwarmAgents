"""Bounded-parallelism worker pool with backpressure.

Learned from asyncio best-practice research (death.andgravity.com et al.):
- Unbounded gather() is "how you DDoS yourself" — gate with a semaphore or
  a worker pool fed by a BOUNDED queue.
- Bounded queues give backpressure: producers block when the queue is full,
  so memory stays constant regardless of task count.
- Per-task timeout kills stragglers; overall budget prevents hangs.

Usage pattern: start the pool (workers drain concurrently) BEFORE or WHILE
submitting; submissions block on the bounded queue, workers keep draining,
so there is no deadlock between a full queue and unstarted workers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PoolStats(BaseModel):
    """Observable pool metrics for tuning max_workers."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    max_concurrency_observed: int = 0
    queue_high_water: int = 0
    elapsed_s: float = 0.0


class WorkerPool:
    """N fixed workers pulling from a bounded queue (constant memory).

    Unlike gather(), submitting 10k items with max_workers=8 keeps at most
    8 coroutines running and queue_max items buffered — no task explosion.
    """

    def __init__(
        self,
        max_workers: int = 8,
        queue_max: int = 64,
        timeout_s: float = 30.0,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers
        self.queue_max = queue_max
        self.timeout_s = timeout_s
        self.stats = PoolStats()
        self._queue: asyncio.Queue[tuple[int, Callable[[], Awaitable[T]]]] = (
            asyncio.Queue(maxsize=queue_max)
        )
        self._results: dict[int, Any] = {}
        self._errors: dict[int, str] = {}
        self._in_flight = 0
        self._worker_tasks: list[asyncio.Task] = []
        self._started = False

    def start(self) -> None:
        """Start workers; must be running before/while submitting many items."""
        if self._started:
            return
        self._started = True
        self._worker_tasks = [
            asyncio.create_task(self._worker()) for _ in range(self.max_workers)
        ]

    async def submit(self, key: int, fn: Callable[[], Awaitable[T]]) -> None:
        """Blocks (backpressure) when the bounded queue is full."""
        if not self._started:
            self.start()
        self.stats.submitted += 1
        await self._queue.put((key, fn))  # awaits when full -> backpressure
        self.stats.queue_high_water = max(
            self.stats.queue_high_water, self._queue.qsize()
        )

    async def _worker(self) -> None:
        while True:
            key, fn = await self._queue.get()
            self._in_flight += 1
            self.stats.max_concurrency_observed = max(
                self.stats.max_concurrency_observed, self._in_flight
            )
            start = time.monotonic()
            try:
                self._results[key] = await asyncio.wait_for(
                    fn(), timeout=self.timeout_s
                )
                self.stats.completed += 1
            except Exception as exc:
                self._errors[key] = f"{type(exc).__name__}: {exc}"
                self.stats.failed += 1
            finally:
                self._in_flight -= 1
                self._queue.task_done()
                self.stats.elapsed_s = max(
                    self.stats.elapsed_s, round(time.monotonic() - start, 4)
                )

    async def run(self) -> tuple[dict[int, Any], dict[int, str]]:
        """Drain queue and return (results, errors). Workers auto-started."""
        self.start()
        await self._queue.join()
        return self._results, self._errors

    async def close(self) -> None:
        for t in self._worker_tasks:
            t.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)


async def bounded_gather(
    fns: list[Callable[[], Awaitable[T]]],
    max_workers: int = 8,
    timeout_s: float = 30.0,
) -> tuple[list[T | None], list[str]]:
    """Semaphore-gated gather for when a full worker pool is overkill."""
    sem = asyncio.Semaphore(max_workers)

    async def one(fn: Callable[[], Awaitable[T]]) -> T | None:
        async with sem:
            try:
                return await asyncio.wait_for(fn(), timeout=timeout_s)
            except Exception:
                return None

    raw = await asyncio.gather(*(one(fn) for fn in fns))
    results = list(raw)
    errors = [f"failed at index {i}" for i, r in enumerate(results) if r is None]
    return results, errors
