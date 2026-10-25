"""Decision-matrix orchestrator tying together all swarm capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from pydanticswarm.affinity import AffinityTracker
from pydanticswarm.communication import Blackboard, Envelope, MessageBus, MessageKind
from pydanticswarm.division import (
    DivisionMode,
    DivisionOutcome,
    divide_multiple,
    divide_same,
)
from pydanticswarm.learning import MemoryStore, Outcome
from pydanticswarm.swarm import AgentResult, AgentSpec, Subtask, Swarm, SwarmOutcome


class Decision(str):
    """Path labels from the decision matrix in Agents/AGENTS.md."""


REJECT = "REJECT"
DIRECT = "DIRECT"
FAN_OUT_SAME = "FAN-OUT-SAME"
FAN_OUT_MULTI = "FAN-OUT-MULTI"
SPECIALIZE = "SPECIALIZE"
AFFINITY_ROUTE = "AFFINITY-ROUTE"


class Orchestrator:
    """Routes a batch of subtasks through the decision matrix.

    Deterministic first: skill availability and division mode decide the
    path. Affinity (learned success rates) only breaks ties — no LLM needed.
    """

    def __init__(
        self,
        agents: Sequence[AgentSpec],
        swarm: Swarm,
        memory: MemoryStore | None = None,
    ) -> None:
        self.agents = list(agents)
        self.swarm = swarm
        self.memory = memory or MemoryStore()
        self.affinity = AffinityTracker(self.memory)
        self.bus = MessageBus()
        self.blackboard = Blackboard()
        for a in self.agents:
            self.bus.register(a.name)

    def decide(self, subtask: Subtask) -> str:
        qualified = [a for a in self.agents if a.can_handle(subtask)]
        if not qualified:
            return REJECT
        if len(qualified) == 1:
            return DIRECT
        if getattr(subtask, "division", None) == DivisionMode.SAME:
            return FAN_OUT_SAME
        if getattr(subtask, "division", None) == DivisionMode.MULTIPLE:
            return FAN_OUT_MULTI
        if len(subtask.required_skills) > 1:
            return SPECIALIZE
        return AFFINITY_ROUTE

    async def run_subtask(self, subtask: Subtask) -> DivisionOutcome | None:
        path = self.decide(subtask)
        if path == REJECT:
            self.memory.record(
                Outcome(
                    agent_name="orchestrator",
                    skill="*",
                    subtask_id=subtask.id,
                    success=False,
                    error="no qualified agent",
                )
            )
            return None
        await self.bus.send(
            Envelope(
                kind=MessageKind.REQUEST,
                sender="orchestrator",
                task_id=subtask.id,
                payload=path,
            )
        )
        if path == FAN_OUT_SAME:
            out = await divide_same(self.agents, subtask, self.swarm.handler)
        elif path == FAN_OUT_MULTI:
            out = await divide_multiple(self.agents, subtask, self.swarm.handler)
        else:
            agent = self.affinity.pick(self.agents, subtask) or self.agents[0]
            self.affinity.started(agent.name)
            try:
                payload = await self.swarm.handler(agent, subtask)
                result = AgentResult(
                    subtask_id=subtask.id, agent_name=agent.name, payload=payload
                )
            except Exception as exc:
                result = AgentResult(
                    subtask_id=subtask.id, agent_name=agent.name, error=str(exc)
                )
            finally:
                self.affinity.finished(agent.name)
            from pydanticswarm.division import DivisionMode as DM

            out = DivisionOutcome(mode=DM.DIFFERENT, results=[result])
        for r in out.results:
            skill = subtask.required_skills[0] if subtask.required_skills else "*"
            self.memory.record(
                Outcome(
                    agent_name=r.agent_name,
                    skill=skill,
                    subtask_id=r.subtask_id,
                    success=r.error is None,
                    payload=r.payload,
                    error=r.error,
                )
            )
        await self.bus.send(
            Envelope(
                kind=MessageKind.RESULT,
                sender="orchestrator",
                task_id=subtask.id,
                payload=out.model_dump(),
            )
        )
        return out

    async def run(self, goal: str, subtasks: Sequence[Subtask]) -> SwarmOutcome:
        outs = await asyncio.gather(*(self.run_subtask(s) for s in subtasks))
        results: list[AgentResult] = []
        for out in outs:
            if out is not None:
                results.extend(out.results)
        return SwarmOutcome(goal=goal, results=results)
