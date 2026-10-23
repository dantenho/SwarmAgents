"""Task division: SAME (vote/race), MULTIPLE (fan-out), DIFFERENT (specialize)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from pydanticswarm.swarm import AgentResult, AgentSpec, Subtask

Handler = Callable[[AgentSpec, Subtask], Awaitable[Any]]


class DivisionMode(str, Enum):
    SAME = "same"
    MULTIPLE = "multiple"
    DIFFERENT = "different"


class Vote(BaseModel):
    agent_name: str
    payload: Any


class DivisionOutcome(BaseModel):
    mode: DivisionMode
    results: list[AgentResult] = Field(default_factory=list)
    winner: Vote | None = None

    @property
    def payload(self) -> Any:
        return self.winner.payload if self.winner else None


async def divide_same(
    agents: Sequence[AgentSpec],
    subtask: Subtask,
    handler: Handler,
    strategy: str = "vote",
) -> DivisionOutcome:
    """Same task to all qualified agents; vote (majority) or race (first wins)."""
    qualified = [a for a in agents if a.can_handle(subtask)]
    if not qualified:
        return DivisionOutcome(mode=DivisionMode.SAME)

    async def one(agent: AgentSpec) -> AgentResult:
        try:
            payload = await handler(agent, subtask)
            return AgentResult(
                subtask_id=subtask.id, agent_name=agent.name, payload=payload
            )
        except Exception as exc:
            return AgentResult(
                subtask_id=subtask.id, agent_name=agent.name, error=str(exc)
            )

    results = await asyncio.gather(*(one(a) for a in qualified))
    valid = [r for r in results if r.error is None]

    if strategy == "race":
        winner = (
            Vote(agent_name=valid[0].agent_name, payload=valid[0].payload)
            if valid
            else None
        )
    else:
        counts: dict[Any, list[str]] = {}
        for r in valid:
            counts.setdefault(r.payload, []).append(r.agent_name)
        best = max(counts.items(), key=lambda kv: len(kv[1]), default=None)
        winner = Vote(agent_name=best[1][0], payload=best[0]) if best else None

    return DivisionOutcome(mode=DivisionMode.SAME, results=list(results), winner=winner)


async def divide_multiple(
    agents: Sequence[AgentSpec],
    subtask: Subtask,
    handler: Handler,
    parts: int = 2,
) -> DivisionOutcome:
    """One task split into N independent slices, fanned out in parallel."""
    slices = [
        Subtask(
            id=f"{subtask.id}#{i}",
            description=f"{subtask.description} [part {i}]",
            required_skills=subtask.required_skills,
        )
        for i in range(parts)
    ]

    async def one(agent: AgentSpec, sub: Subtask) -> AgentResult:
        try:
            payload = await handler(agent, sub)
            return AgentResult(
                subtask_id=sub.id, agent_name=agent.name, payload=payload
            )
        except Exception as exc:
            return AgentResult(subtask_id=sub.id, agent_name=agent.name, error=str(exc))

    tasks = []
    for sub in slices:
        qualified = [a for a in agents if a.can_handle(sub)]
        if not qualified:
            continue
        tasks.append(one(qualified[0], sub))
    results = await asyncio.gather(*tasks) if tasks else []
    return DivisionOutcome(mode=DivisionMode.MULTIPLE, results=list(results))


async def divide_different(
    agents: Sequence[AgentSpec],
    subtasks: Sequence[Subtask],
    handler: Handler,
) -> DivisionOutcome:
    """Unrelated tasks routed by skill, all run concurrently."""

    async def one(agent: AgentSpec, sub: Subtask) -> AgentResult:
        try:
            payload = await handler(agent, sub)
            return AgentResult(
                subtask_id=sub.id, agent_name=agent.name, payload=payload
            )
        except Exception as exc:
            return AgentResult(subtask_id=sub.id, agent_name=agent.name, error=str(exc))

    pairs = []
    for sub in subtasks:
        qualified = [a for a in agents if a.can_handle(sub)]
        if qualified:
            pairs.append(one(qualified[0], sub))
    results = await asyncio.gather(*pairs) if pairs else []
    return DivisionOutcome(mode=DivisionMode.DIFFERENT, results=list(results))
