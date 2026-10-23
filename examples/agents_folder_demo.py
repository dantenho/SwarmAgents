"""End-to-end demo: decision matrix, division modes, affinity and learning.

Run: pixi run python examples/agents_folder_demo.py
"""

import asyncio

from pydanticswarm import (
    AffinityTracker,
    AgentSpec,
    DivisionMode,
    MemoryStore,
    Orchestrator,
    Outcome,
    Subtask,
    Swarm,
)


async def handler(agent: AgentSpec, subtask: Subtask) -> str:
    if agent.name == "flaky-coder":
        raise RuntimeError("flaky failure")
    return f"[{agent.name}] {subtask.description}: done"


async def main() -> None:
    memory = MemoryStore()
    for _ in range(3):
        memory.record(
            Outcome(agent_name="coder", skill="code", subtask_id="warm", success=True)
        )

    agents = [
        AgentSpec(name="coder", skills=["code"]),
        AgentSpec(name="flaky-coder", skills=["code"]),
        AgentSpec(name="writer", skills=["write"]),
        AgentSpec(name="reviewer-a", skills=["review"]),
        AgentSpec(name="reviewer-b", skills=["review"]),
    ]
    swarm = Swarm(agents=agents, handler=handler)
    orch = Orchestrator(agents=agents, swarm=swarm, memory=memory)

    print("== Decision matrix paths ==")
    cases = [
        ("no skill match", Subtask(id="1", description="d", required_skills=["zzz"])),
        (
            "direct (solo writer)",
            Subtask(id="2", description="write doc", required_skills=["write"]),
        ),
        (
            "fan-out same (consensus review)",
            Subtask(
                id="3",
                description="review PR",
                required_skills=["review"],
                division=DivisionMode.SAME,
            ),
        ),
        (
            "fan-out multi (split work)",
            Subtask(
                id="4",
                description="refactor module",
                required_skills=["code"],
                division=DivisionMode.MULTIPLE,
            ),
        ),
    ]
    for label, sub in cases:
        print(f"  {label:<32} -> {orch.decide(sub)}")

    print("\n== Affinity picks the proven agent ==")
    tracker = AffinityTracker(memory)
    picked = tracker.pick(
        [agents[1], agents[0]],
        Subtask(id="x", description="code it", required_skills=["code"]),
    )
    print(f"  picked: {picked.name} (coder has 100% success history)")

    print("\n== Full run (flaky agent fails, outcomes recorded) ==")
    outcome = await orch.run(
        goal="ship feature",
        subtasks=[
            Subtask(id="a", description="write docs", required_skills=["write"]),
            Subtask(id="b", description="code feature", required_skills=["code"]),
            Subtask(
                id="c",
                description="review result",
                required_skills=["review"],
                division=DivisionMode.SAME,
            ),
        ],
    )
    for r in outcome.results:
        status = "OK " if r.error is None else "ERR"
        line = f"  [{status}] {r.subtask_id:<3} {r.agent_name:<12}"
        print(f"{line} payload={r.payload} err={r.error}")

    print("\n== Learned outcomes (self-learning fast loop) ==")
    for o in orch.memory.outcomes:
        print(f"  {o.agent_name:<12} skill={o.skill:<7} success={o.success}")

    print("\n== Message bus traffic ==")
    kinds = [e.kind.value for e in orch.bus.log]
    print(f"  {len(kinds)} envelopes: {set(kinds)}")

    print("\n== Blackboard ==")
    orch.blackboard.write("goal", "ship feature")
    print(f"  goal = {orch.blackboard.read('goal')}")


if __name__ == "__main__":
    asyncio.run(main())
