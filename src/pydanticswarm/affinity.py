"""Affinity router: learned per-(agent, skill) success scoring."""

from __future__ import annotations

from pydanticswarm.learning import MemoryStore
from pydanticswarm.swarm import AgentSpec, Subtask


class AffinityTracker:
    """score = skill_coverage + success_rate - failure_penalty - busyness."""

    def __init__(self, memory: MemoryStore, decay: float = 0.9) -> None:
        self.memory = memory
        self.decay = decay
        self.in_flight: dict[str, int] = {}

    def score(self, agent: AgentSpec, subtask: Subtask) -> float:
        if not agent.can_handle(subtask):
            return -1.0
        coverage = len(subtask.required_skills) / max(len(agent.skills), 1)
        rate = (
            self.memory.success_rate(agent.name, subtask.required_skills[0])
            if subtask.required_skills
            else 0.5
        )
        busy = self.in_flight.get(agent.name, 0)
        return coverage + rate - 0.1 * busy

    def pick(self, candidates: list[AgentSpec], subtask: Subtask) -> AgentSpec | None:
        qualified = [a for a in candidates if a.can_handle(subtask)]
        if not qualified:
            return None
        return max(qualified, key=lambda a: self.score(a, subtask))

    def started(self, agent_name: str) -> None:
        self.in_flight[agent_name] = self.in_flight.get(agent_name, 0) + 1

    def finished(self, agent_name: str) -> None:
        n = self.in_flight.get(agent_name, 0) - 1
        self.in_flight[agent_name] = max(n, 0)
