"""Core parallel orchestration primitives for SwarmAgents."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

from swarmagents.metrics import MetricsRegistry

T = TypeVar("T")
R = TypeVar("R")

Reporter = Callable[[str, dict[str, Any]], None]


class Swarm:
    """Run many agents concurrently with bounded parallelism and metrics.

    An agent is any callable (sync or async) that accepts one input item
    and returns a result. Execution is capped at ``max_concurrency``
    in-flight agents, so a large swarm does not exhaust resources.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 10,
        timeout: float | None = None,
        registry: MetricsRegistry | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.registry = registry or MetricsRegistry()
        self.reporter = reporter

    async def run(
        self, agents: Iterable[Callable[[T], R]], items: Iterable[T]
    ) -> list[R]:
        """Run each agent against its paired item concurrently.

        Returns results in input order. Failures are recorded as metrics
        and re-raised after all agents settle.
        """
        pairs = list(zip(agents, items, strict=True))
        self.registry.increment("swarm.submitted", value=len(pairs))
        start = time.perf_counter()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        sentinel: Any = object()
        results: list[Any] = [sentinel] * len(pairs)
        errors: list[BaseException | None] = [None] * len(pairs)

        async def run_one(index: int, agent: Callable[[T], R], item: T) -> None:
            async with semaphore:
                agent_start = time.perf_counter()
                try:
                    if asyncio.iscoroutinefunction(agent):
                        coro: Awaitable[R] = agent(item)
                    else:
                        coro = asyncio.to_thread(agent, item)
                    if self.timeout is not None:
                        result = await asyncio.wait_for(coro, timeout=self.timeout)
                    else:
                        result = await coro
                    results[index] = result
                    self.registry.increment("swarm.success")
                except BaseException as exc:
                    errors[index] = exc
                    self.registry.increment("swarm.failure")
                finally:
                    elapsed = time.perf_counter() - agent_start
                    self.registry.observe("swarm.agent_duration_seconds", elapsed)
                    if self.reporter is not None:
                        self.reporter(
                            "swarm.agent",
                            {"index": index, "duration_seconds": elapsed},
                        )

        await asyncio.gather(*(run_one(i, a, it) for i, (a, it) in enumerate(pairs)))
        elapsed = time.perf_counter() - start
        self.registry.observe("swarm.total_duration_seconds", elapsed)
        for exc in errors:
            if exc is not None:
                raise exc
        return [r for r in results if r is not sentinel]

    def run_sync(
        self, agents: Iterable[Callable[[T], R]], items: Iterable[T]
    ) -> list[R]:
        return asyncio.run(self.run(agents, items))
