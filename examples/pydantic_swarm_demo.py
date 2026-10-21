"""Example: PydanticSwarm in action.

Run with: pixi run python examples/pydantic_swarm_demo.py
"""

import asyncio

from pydanticswarm import AgentSpec, Subtask, Swarm


async def research_handler(agent: AgentSpec, subtask: Subtask):
    await asyncio.sleep(0.1)
    return f"[{agent.name}] research notes for: {subtask.description}"


async def code_handler(agent: AgentSpec, subtask: Subtask):
    await asyncio.sleep(0.1)
    return f"[{agent.name}] implementation for: {subtask.description}"


async def write_handler(agent: AgentSpec, subtask: Subtask):
    await asyncio.sleep(0.1)
    return f"[{agent.name}] draft for: {subtask.description}"


HANDLERS = {
    "researcher": research_handler,
    "coder": code_handler,
    "writer": write_handler,
}


async def routed_handler(agent: AgentSpec, subtask: Subtask):
    return await HANDLERS[agent.name](agent, subtask)


async def main() -> None:
    agents = [
        AgentSpec(name="researcher", skills=["search", "summarize"]),
        AgentSpec(name="coder", skills=["code", "test"]),
        AgentSpec(name="writer", skills=["write", "edit"]),
    ]

    swarm = Swarm(agents=agents, handler=routed_handler)

    outcome = await swarm.run(
        goal="Publish a blog post about Pydantic swarms",
        plan=[
            Subtask(
                id="research",
                description="collect Pydantic v2 features",
                required_skills=["search"],
            ),
            Subtask(
                id="examples",
                description="write runnable code examples",
                required_skills=["code"],
                depends_on=["research"],
            ),
            Subtask(
                id="draft",
                description="compose the final post",
                required_skills=["write"],
                depends_on=["research", "examples"],
            ),
        ],
    )

    print(outcome.summary)
    for r in outcome.results:
        print(f"  {r.subtask_id:<10} by {r.agent_name:<12} ({r.elapsed_s}s)")


if __name__ == "__main__":
    asyncio.run(main())
