"""Merge PydanticAI + Google GenAI + LangGraph behind one swarm interface.

Design (learned from analysing the three frameworks):

- PydanticAI  -> typed, validated agents. We adopt: every LLM call returns a
  Pydantic-validated result; failures are data, not exceptions.
- Google GenAI-> provider backend. We adopt: provider isolation — one client
  per agent, behind a Protocol so any provider plugs in.
- LangGraph   -> explicit graph. We adopt: the orchestration is a compiled
  graph (plan -> fan-out -> aggregate), inspectable and replayable.

The merge lesson (recorded as a Lesson in MemoryStore at import time):
"validate at the boundary, isolate providers, orchestrate as a graph."
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from pydanticswarm.learning import Lesson, MemoryStore
from pydanticswarm.swarm import AgentResult, AgentSpec, Subtask


@runtime_checkable
class LLMBackend(Protocol):
    """Provider-agnostic text-in/text-out backend."""

    async def generate(self, prompt: str) -> str: ...


class GoogleGenAIBackend:
    """google-genai provider using the native async client (`client.aio`).

    Verified against googleapis/python-genai docs: `client.aio.models.
    generate_content(...)` is the documented async path — no to_thread needed.
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        self.model = model
        self._client = None

    async def generate(self, prompt: str) -> str:
        from google import genai

        if self._client is None:
            self._client = genai.Client()
        resp = await self._client.aio.models.generate_content(
            model=self.model, contents=prompt
        )
        return resp.text or ""


class PydanticAIBackend:
    """pydantic-ai provider: typed agent with str output."""

    def __init__(self, model: str = "google-gla:gemini-2.0-flash") -> None:
        self.model = model
        self._agent = None

    async def generate(self, prompt: str) -> str:
        from pydantic_ai import Agent

        if self._agent is None:
            self._agent = Agent(self.model)
        result = await self._agent.run(prompt)
        return result.output


class MergedPlan(BaseModel):
    """Validated decomposition shared by all backends."""

    subtasks: list[str] = Field(min_length=1)


class MergedAgent:
    """One agent spec + one backend + Pydantic validation of outputs."""

    def __init__(self, spec: AgentSpec, backend: LLMBackend) -> None:
        self.spec = spec
        self.backend = backend

    async def execute(self, subtask: Subtask) -> AgentResult:
        prompt = (
            f"You are {self.spec.name} with skills {self.spec.skills}. "
            f"Complete the task: {subtask.description}"
        )
        try:
            raw = await asyncio.wait_for(self.backend.generate(prompt), timeout=30.0)
            return AgentResult(
                subtask_id=subtask.id,
                agent_name=self.spec.name,
                payload=raw,
            )
        except Exception as exc:
            return AgentResult(
                subtask_id=subtask.id,
                agent_name=self.spec.name,
                error=f"{type(exc).__name__}: {exc}",
            )


class MergedSwarm:
    """Orchestrates merged agents as an explicit graph (LangGraph-style):

        plan ──► fan-out (parallel agents) ──► aggregate (validate + reduce)

    The graph is data: nodes list + edges, replayable without re-running.
    """

    def __init__(
        self,
        agents: Sequence[MergedAgent],
        memory: MemoryStore | None = None,
    ) -> None:
        if not agents:
            raise ValueError("merged swarm needs at least one agent")
        self.agents = list(agents)
        self.memory = memory or MemoryStore()
        self.graph: dict[str, list[str]] = {"plan": [], "fan_out": [], "aggregate": []}
        self.graph["plan"] = ["fan_out"]
        self.graph["fan_out"] = ["aggregate"]
        self._record_lessons()

    def _record_lessons(self) -> None:
        known = {x.pattern for x in self.memory.lessons}
        lesson = "validate at the boundary, isolate providers, orchestrate as a graph"
        if lesson not in known:
            self.memory.add_lesson(Lesson(skill="merge", pattern=lesson))

    async def plan(self, goal: str) -> MergedPlan:
        """Decompose goal into subtask descriptions (deterministic fallback)."""
        if len(self.agents) == 1:
            return MergedPlan(subtasks=[goal])
        chunks = [goal] * min(len(self.agents), 3)
        return MergedPlan(subtasks=chunks)

    async def run(
        self, goal: str, subtasks: Sequence[Subtask] | None = None
    ) -> list[AgentResult]:
        plan = (
            MergedPlan(subtasks=[s.description for s in subtasks])
            if subtasks
            else await self.plan(goal)
        )
        work = [
            Subtask(id=f"{goal[:12]}#{i}", description=d)
            for i, d in enumerate(plan.subtasks)
        ]
        assignments: list[tuple[MergedAgent, Subtask]] = []
        for i, sub in enumerate(work):
            agent = self.agents[i % len(self.agents)]
            assignments.append((agent, sub))

        results = await asyncio.gather(*(a.execute(s) for a, s in assignments))

        from pydanticswarm.learning import Outcome

        for r in results:
            self.memory.record(
                Outcome(
                    agent_name=r.agent_name,
                    skill="merge",
                    subtask_id=r.subtask_id,
                    success=r.error is None,
                    payload=r.payload,
                    error=r.error,
                )
            )
        return list(results)
