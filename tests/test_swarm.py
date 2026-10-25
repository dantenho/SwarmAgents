import asyncio

import pytest

from pydanticswarm import AgentResult, AgentSpec, Subtask, Swarm


def test_agent_skill_matching():
    agent = AgentSpec(name="coder", skills=["code", "test"])
    assert agent.can_handle(Subtask(id="1", description="d", required_skills=["code"]))
    assert not agent.can_handle(
        Subtask(id="2", description="d", required_skills=["search"])
    )


def test_swarm_requires_agents():
    with pytest.raises(ValueError):
        Swarm(agents=[])


def test_swarm_runs_plan_and_aggregates():
    swarm = Swarm(
        agents=[
            AgentSpec(name="researcher", skills=["search"]),
            AgentSpec(name="coder", skills=["code"]),
        ]
    )
    outcome = asyncio.run(
        swarm.run(
            goal="demo",
            plan=[
                Subtask(id="1", description="find libs", required_skills=["search"]),
                Subtask(id="2", description="write code", required_skills=["code"]),
            ],
        )
    )
    assert len(outcome.results) == 2
    assert not outcome.failures
    assert "complete" in outcome.summary


def test_dependency_ordering():
    swarm = Swarm(agents=[AgentSpec(name="a", skills=["x"])])
    outcome = asyncio.run(
        swarm.run(
            goal="demo",
            plan=[
                Subtask(id="1", description="first", required_skills=["x"]),
                Subtask(
                    id="2",
                    description="second",
                    required_skills=["x"],
                    depends_on=["1"],
                ),
            ],
        )
    )
    assert len(outcome.results) == 2


def test_failure_isolated_and_retried():
    calls = {"n": 0}

    async def flaky(agent, sub):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("boom")
        return "ok"

    swarm = Swarm(agents=[AgentSpec(name="w", skills=["x"])], handler=flaky)
    outcome = asyncio.run(
        swarm.run(
            goal="demo",
            plan=[
                Subtask(id="1", description="job", required_skills=["x"], max_retries=2)
            ],
        )
    )
    assert len(outcome.results) == 1
    assert outcome.results[0].payload == "ok"


def test_agent_result_model():
    r = AgentResult(subtask_id="1", agent_name="w", payload={"k": 1})
    assert r.error is None
    assert r.elapsed_s == 0.0
