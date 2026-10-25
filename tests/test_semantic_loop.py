"""End-to-end: lesson learned -> semantic retrieval -> prompt injection."""

import asyncio

from pydanticswarm import (
    AgentSpec,
    MergedAgent,
    MergedSwarm,
    SemanticMemoryStore,
    Subtask,
)
from pydanticswarm.learning import Lesson


class StubBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "ok"


def test_semantic_loop_injects_lessons_into_prompt():
    async def flow():
        mem = SemanticMemoryStore()
        mem.add_lesson(
            Lesson(
                skill="code",
                pattern="retry with smaller prompt",
                context="llm code gen",
            )
        )
        mem.add_lesson(
            Lesson(skill="write", pattern="outline first", context="blog writing")
        )
        backend = StubBackend()
        agent = MergedAgent(AgentSpec(name="coder", skills=["code"]), backend)
        swarm = MergedSwarm(agents=[agent], memory=mem)
        await swarm.run(
            "goal",
            subtasks=[Subtask(id="1", description="llm code generation task")],
        )
        assert len(backend.prompts) == 1
        # the code-gen lesson is semantically closest -> must appear in prompt
        assert "retry with smaller prompt" in backend.prompts[0]
        # the unrelated writing lesson must NOT be injected
        assert "outline first" not in backend.prompts[0]

    asyncio.run(flow())
