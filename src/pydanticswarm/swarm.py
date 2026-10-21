"""Core swarm primitives: agents, subtasks, dispatcher and aggregation.

Pure Pydantic models + asyncio. No LLM framework required.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

Handler = Callable[["AgentSpec", "Subtask"], Awaitable[Any]]


class AgentSpec(BaseModel):
    """Declarative description of a swarm agent."""

    name: str
    skills: list[str] = Field(default_factory=list)
    model: str = "default"
    max_concurrency: int = Field(default=1, ge=1)

    def can_handle(self, subtask: "Subtask") -> bool:
        return set(subtask.required_skills).issubset(set(self.skills))


class Subtask(BaseModel):
    """A unit of work produced by the planner."""

    id: str
    description: str
    required_skills: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    max_retries: int = Field(default=2, ge=0)
    _retries: int = 0

    def record_retry(self) -> None:
        self._retries += 1

    @property
    def exhausted(self) -> bool:
        return self._retries >= self.max_retries


class AgentResult(BaseModel):
    """Validated output of one subtask, produced by one agent."""

    subtask_id: str
    agent_name: str
    payload: Any = None
    elapsed_s: float = 0.0
    error: str | None = None


class SwarmOutcome(BaseModel):
    """Final reduced state of a swarm run."""

    goal: str
    results: list[AgentResult] = Field(default_factory=list)
    failures: list[AgentResult] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [r.payload for r in self.results if r.payload is not None]
        joined = "\n".join(str(p) for p in parts)
        status = "complete" if not self.failures else "partial"
        return f"[{status}] goal={self.goal!r}\n{joined}"


DefaultHandler = Callable[[AgentSpec, Subtask], Awaitable[Any]]


async def _default_handler(agent: AgentSpec, subtask: Subtask) -> Any:
    return f"[{agent.name}] {subtask.description}: done"


class Swarm:
    """In-memory swarm: planner output in, parallel agents, reduced output.

    The dispatcher is capacity-aware: a worker only picks a subtask when
    the agent has a matching skill and a free concurrency slot.
    """

    def __init__(
        self,
        agents: Sequence[AgentSpec],
        handler: Handler | None = None,
        max_parallel: int | None = None,
    ) -> None:
        if not agents:
            raise ValueError("swarm needs at least one agent")
        self.agents = list(agents)
        self.handler: DefaultHandler = handler or _default_handler
        self.max_parallel = max_parallel or len(self.agents)

    async def run(self, goal: str, plan: Sequence[Subtask]) -> SwarmOutcome:
        pending = {s.id: s for s in plan}
        done: dict[str, AgentResult] = {}
        failures: list[AgentResult] = []
        in_flight = 0

        for attempt_cycle in range(self.max_retries_global(plan)):
            ready = [
                s for s in pending.values() if all(dep in done for dep in s.depends_on)
            ]
            if not ready and not in_flight:
                break

            queue: asyncio.Queue[Subtask] = asyncio.Queue()
            for s in ready:
                queue.put_nowait(s)

            results = await self._execute(ready, queue, done)
            for r in results:
                if r.error is None:
                    done[r.subtask_id] = r
                    pending.pop(r.subtask_id, None)
                else:
                    sub = pending.get(r.subtask_id)
                    if sub is not None and not sub.exhausted:
                        sub.record_retry()
                    else:
                        failures.append(r)
                        pending.pop(r.subtask_id, None)

            if not pending:
                break

        return SwarmOutcome(goal=goal, results=list(done.values()), failures=failures)

    def max_retries_global(self, plan: Sequence[Subtask]) -> int:
        return 1 + max((s.max_retries for s in plan), default=0)

    async def _execute(
        self,
        ready: Sequence[Subtask],
        queue: asyncio.Queue,
        done: dict[str, AgentResult],
    ) -> list[AgentResult]:
        sem = asyncio.Semaphore(self.max_parallel)
        tasks: list[asyncio.Task] = []

        async def worker(agent: AgentSpec) -> None:
            while True:
                try:
                    sub = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                if not agent.can_handle(sub):
                    queue.put_nowait(sub)
                    if all(not a.can_handle(sub) for a in self.agents):
                        return
                    await asyncio.sleep(0)
                    continue
                async with sem:
                    task = asyncio.create_task(self._run_one(agent, sub))
                    tasks.append(task)

        # start one coroutine per agent; each pulls from the shared queue
        await asyncio.gather(*(worker(a) for a in self.agents))
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[AgentResult] = []
        for g in gathered:
            if isinstance(g, AgentResult):
                out.append(g)
            elif isinstance(g, BaseException):
                out.append(
                    AgentResult(subtask_id="unknown", agent_name="swarm", error=str(g))
                )
        return out

    async def _run_one(self, agent: AgentSpec, sub: Subtask) -> AgentResult:
        if not agent.can_handle(sub):
            return AgentResult(
                subtask_id=sub.id,
                agent_name=agent.name,
                error=f"agent lacks skills {sub.required_skills}",
            )
        start = time.monotonic()
        try:
            payload = await asyncio.wait_for(self.handler(agent, sub), timeout=60.0)
            return AgentResult(
                subtask_id=sub.id,
                agent_name=agent.name,
                payload=payload,
                elapsed_s=round(time.monotonic() - start, 4),
            )
        except Exception as exc:
            return AgentResult(
                subtask_id=sub.id,
                agent_name=agent.name,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=round(time.monotonic() - start, 4),
            )
