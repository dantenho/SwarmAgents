import asyncio

from pydanticswarm import AgentSpec, MemoryStore, Subtask
from pydanticswarm.merge import (
    GoogleGenAIBackend,
    LLMBackend,
    MergedAgent,
    MergedPlan,
    MergedSwarm,
    PydanticAIBackend,
)


class FakeBackend:
    """Deterministic backend for tests — no network, no API key."""

    def __init__(self, prefix: str = "fake") -> None:
        self.prefix = prefix
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        return f"{self.prefix}:{prompt[-40:]}"


def test_lessons_recorded_on_merge():
    mem = MemoryStore()
    MergedSwarm(
        agents=[MergedAgent(AgentSpec(name="a", skills=["x"]), FakeBackend())],
        memory=mem,
    )
    patterns = [x.pattern for x in mem.lessons_for("merge")]
    assert (
        "validate at the boundary, isolate providers, orchestrate as a graph"
        in patterns
    )


def test_merged_swarm_requires_agents():
    try:
        MergedSwarm(agents=[])
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "at least one agent" in str(e)


def test_graph_structure_is_explicit():
    swarm = MergedSwarm(
        agents=[MergedAgent(AgentSpec(name="a", skills=["x"]), FakeBackend())]
    )
    assert swarm.graph == {
        "plan": ["fan_out"],
        "fan_out": ["aggregate"],
        "aggregate": [],
    }
    assert swarm.graph == {
        "plan": ["fan_out"],
        "fan_out": ["aggregate"],
        "aggregate": [],
    }


def test_backends_satisfy_protocol():
    assert isinstance(GoogleGenAIBackend(), LLMBackend)
    assert isinstance(PydanticAIBackend(), LLMBackend)
    assert isinstance(FakeBackend(), LLMBackend)


def test_merged_run_round_robins_and_records_outcomes():
    mem = MemoryStore()
    b1, b2 = FakeBackend("b1"), FakeBackend("b2")
    swarm = MergedSwarm(
        agents=[
            MergedAgent(AgentSpec(name="a", skills=["x"]), b1),
            MergedAgent(AgentSpec(name="b", skills=["x"]), b2),
        ],
        memory=mem,
    )
    results = asyncio.run(
        swarm.run(
            "do stuff",
            subtasks=[
                Subtask(id="1", description="part one"),
                Subtask(id="2", description="part two"),
            ],
        )
    )
    assert len(results) == 2
    assert b1.calls == 1 and b2.calls == 1
    assert all(r.error is None for r in results)
    assert len(mem.outcomes) == 2


def test_failure_isolated_as_data():
    class Boom:
        async def generate(self, prompt: str) -> str:
            raise RuntimeError("backend down")

    mem = MemoryStore()
    swarm = MergedSwarm(
        agents=[MergedAgent(AgentSpec(name="a", skills=["x"]), Boom())],
        memory=mem,
    )
    results = asyncio.run(
        swarm.run("solo", subtasks=[Subtask(id="1", description="d")])
    )
    assert results[0].error is not None
    assert mem.outcomes[0].success is False


def test_plan_is_validated():
    swarm = MergedSwarm(
        agents=[MergedAgent(AgentSpec(name="a", skills=["x"]), FakeBackend())]
    )
    plan = asyncio.run(swarm.plan("single goal"))
    assert isinstance(plan, MergedPlan)
    assert plan.subtasks == ["single goal"]
