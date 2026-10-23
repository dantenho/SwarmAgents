import asyncio

from pydanticswarm import AgentSpec, Subtask, Swarm
from pydanticswarm.affinity import AffinityTracker
from pydanticswarm.communication import Blackboard, Envelope, MessageBus, MessageKind
from pydanticswarm.division import (
    DivisionMode,
    divide_different,
    divide_multiple,
    divide_same,
)
from pydanticswarm.learning import Lesson, MemoryStore, Outcome
from pydanticswarm.orchestrator import (
    AFFINITY_ROUTE,
    DIRECT,
    FAN_OUT_MULTI,
    FAN_OUT_SAME,
    REJECT,
    SPECIALIZE,
    Orchestrator,
)


async def ok_handler(agent, subtask):
    return f"[{agent.name}] {subtask.description}"


def test_decision_matrix_paths():
    agents = [
        AgentSpec(name="solo", skills=["a"]),
        AgentSpec(name="multi1", skills=["b"]),
        AgentSpec(name="multi2", skills=["b"]),
        AgentSpec(name="both1", skills=["b", "c"]),
        AgentSpec(name="both2", skills=["b", "c"]),
    ]
    swarm = Swarm(agents=agents, handler=ok_handler)
    orch = Orchestrator(agents=agents, swarm=swarm)

    assert (
        orch.decide(Subtask(id="1", description="d", required_skills=["zzz"])) == REJECT
    )
    assert (
        orch.decide(Subtask(id="2", description="d", required_skills=["a"])) == DIRECT
    )
    sub_same = Subtask(
        id="3", description="d", required_skills=["b"], division=DivisionMode.SAME
    )
    assert orch.decide(sub_same) == FAN_OUT_SAME
    sub_multi = Subtask(
        id="4", description="d", required_skills=["b"], division=DivisionMode.MULTIPLE
    )
    assert orch.decide(sub_multi) == FAN_OUT_MULTI
    assert (
        orch.decide(Subtask(id="5", description="d", required_skills=["b", "c"]))
        == SPECIALIZE
    )


def test_decision_matrix_affinity_route():
    agents = [AgentSpec(name="x", skills=["b"]), AgentSpec(name="y", skills=["b"])]
    swarm = Swarm(agents=agents, handler=ok_handler)
    orch = Orchestrator(agents=agents, swarm=swarm)
    assert (
        orch.decide(Subtask(id="1", description="d", required_skills=["b"]))
        == AFFINITY_ROUTE
    )


def test_divide_same_vote():
    async def handler(agent, subtask):
        return "consensus"

    agents = [AgentSpec(name="a", skills=["s"]), AgentSpec(name="b", skills=["s"])]
    sub = Subtask(
        id="1", description="d", required_skills=["s"], division=DivisionMode.SAME
    )
    out = asyncio.run(divide_same(agents, sub, handler, strategy="vote"))
    assert out.winner is not None
    assert out.winner.payload == "consensus"
    assert len(out.results) == 2


def test_divide_same_race():
    async def handler(agent, subtask):
        return f"from-{agent.name}"

    agents = [AgentSpec(name="a", skills=["s"]), AgentSpec(name="b", skills=["s"])]
    sub = Subtask(
        id="1", description="d", required_skills=["s"], division=DivisionMode.SAME
    )
    out = asyncio.run(divide_same(agents, sub, handler, strategy="race"))
    assert out.winner is not None
    assert out.winner.payload.startswith("from-")


def test_divide_multiple_fanout():
    agents = [AgentSpec(name="a", skills=["s"])]
    sub = Subtask(
        id="1",
        description="work",
        required_skills=["s"],
        division=DivisionMode.MULTIPLE,
    )
    out = asyncio.run(divide_multiple(agents, sub, ok_handler, parts=3))
    assert len(out.results) == 3
    assert all(r.payload for r in out.results)


def test_divide_different():
    agents = [
        AgentSpec(name="coder", skills=["code"]),
        AgentSpec(name="writer", skills=["write"]),
    ]
    subs = [
        Subtask(id="1", description="code it", required_skills=["code"]),
        Subtask(id="2", description="write it", required_skills=["write"]),
    ]
    out = asyncio.run(divide_different(agents, subs, ok_handler))
    assert len(out.results) == 2


def test_affinity_prefers_successful_agent():
    mem = MemoryStore()
    for _ in range(3):
        mem.record(Outcome(agent_name="good", skill="s", subtask_id="x", success=True))
    mem.record(Outcome(agent_name="bad", skill="s", subtask_id="y", success=False))
    tracker = AffinityTracker(mem)
    good = AgentSpec(name="good", skills=["s"])
    bad = AgentSpec(name="bad", skills=["s"])
    sub = Subtask(id="1", description="d", required_skills=["s"])
    assert tracker.pick([bad, good], sub).name == "good"


def test_message_bus_and_blackboard():
    async def flow():
        bus = MessageBus()
        bus.register("a")
        await bus.send(
            Envelope(
                kind=MessageKind.REQUEST, sender="orch", recipient="a", payload="hi"
            )
        )
        env = await bus.receive("a", timeout=1)
        assert env.payload == "hi"
        assert len(bus.log) == 1

    asyncio.run(flow())

    bb = Blackboard()
    bb.write("key", "value")
    assert bb.read("key") == "value"
    assert len(bb.history) == 1


def test_memory_lessons(tmp_path):
    mem = MemoryStore(persist_path=tmp_path / "memory.json")
    mem.record(Outcome(agent_name="a", skill="s", subtask_id="1", success=True))
    mem.add_lesson(Lesson(skill="s", pattern="retry with simpler prompt"))
    mem2 = MemoryStore(persist_path=tmp_path / "memory.json")
    assert len(mem2.outcomes) == 1
    assert mem2.lessons_for("s")[0].pattern == "retry with simpler prompt"
    assert mem2.success_rate("a", "s") == 1.0


def test_orchestrator_end_to_end_records_outcomes():
    agents = [
        AgentSpec(name="r", skills=["search"]),
        AgentSpec(name="c", skills=["code"]),
    ]
    swarm = Swarm(agents=agents, handler=ok_handler)
    orch = Orchestrator(agents=agents, swarm=swarm)
    outcome = asyncio.run(
        orch.run(
            goal="demo",
            subtasks=[
                Subtask(id="1", description="search", required_skills=["search"]),
                Subtask(id="2", description="code", required_skills=["code"]),
            ],
        )
    )
    assert len(outcome.results) == 2
    assert len(orch.memory.outcomes) == 2
