"""Tests for the swarm package (no API keys or network required)."""

from __future__ import annotations

import pytest

import swarmagents
from swarmagents.providers import (
    PROVIDER_NAMES,
    ProviderError,
    get_chat_model,
    get_provider_spec,
)
from swarmagents.swarm import (
    AGENT_ROLES,
    create_langgraph_swarm,
)


def test_registry_has_20_providers():
    assert len(PROVIDER_NAMES) == 20
    assert len(set(PROVIDER_NAMES)) == 20


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_every_provider_resolves(name):
    spec = get_provider_spec(name)
    assert spec.name == name
    assert spec.default_model


def test_provider_lookup_is_case_insensitive():
    assert get_provider_spec("OpenAI").name == "openai"


def test_unknown_provider_raises():
    with pytest.raises(ProviderError):
        get_provider_spec("not-a-provider")


def test_openai_compatible_providers_have_base_url():
    spec = get_provider_spec("openrouter")
    assert spec.base_url is not None


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        get_chat_model("openai")


def test_swarm_agent_roles_defined():
    assert set(AGENT_ROLES) == {"researcher", "analyst", "writer"}


def test_package_exports():
    assert hasattr(swarmagents, "create_langgraph_swarm")
    assert hasattr(swarmagents, "run_langgraph_swarm")
    assert hasattr(swarmagents, "run_swarm")


def test_swarm_builds_with_fake_model(monkeypatch):
    class FakeModel:
        def bind_tools(self, tools, **kwargs):
            return self

    monkeypatch.setattr("swarmagents.swarm.get_chat_model", lambda *a, **k: FakeModel())
    swarm = create_langgraph_swarm(provider="openai")
    assert swarm is not None


def test_vectordb_local_roundtrip(monkeypatch):
    from swarmagents.vectordb import LocalVectorStore

    monkeypatch.setattr(
        "swarmagents.vectordb.embed",
        lambda t, use_cache=True: [ord(c) % 127 for c in t.ljust(384, " ")],
    )
    store = LocalVectorStore()
    ids = store.add(["alpha doc", "beta doc"], [{"n": 1}, {"n": 2}])
    assert ids == ["0", "1"]
    results = store.search("alpha doc", k=1)
    assert len(results) == 1
    text, score, meta = results[0]
    assert text == "alpha doc" and score == 1.0 and meta == {"n": 1}


def test_create_vector_store_falls_back_local(monkeypatch):
    from swarmagents.vectordb import LocalVectorStore, create_vector_store

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert isinstance(create_vector_store(), LocalVectorStore)


def test_rust_batch_matches_single():
    from swarmagents_core import cosine_batch_int8, cosine_similarity_int8

    a = [1, 2, 3, 4]
    b = [4, 3, 2, 1]
    batch = cosine_batch_int8([a, a], [b, a])
    assert abs(batch[0] - cosine_similarity_int8(a, b)) < 1e-12
    assert abs(batch[1] - 1.0) < 1e-12


def test_rust_batch_misaligned_raises():
    import pytest
    from swarmagents_core import cosine_batch_int8

    with pytest.raises(ValueError):
        cosine_batch_int8([[1]], [[1], [2]])


def test_tracing_disabled_without_env(monkeypatch):
    from swarmagents.tracing import get_tracer, tracing_enabled

    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert not tracing_enabled()
    assert get_tracer() is None


def test_tracing_enabled_builds_tracer(monkeypatch):
    from swarmagents.tracing import get_tracer, tracing_enabled

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "swarmagents-test")
    assert tracing_enabled()
    tracer = get_tracer()
    assert tracer is not None
    assert getattr(tracer, "project_name", None) == "swarmagents-test"


def test_traced_decorator_passthrough():
    from swarmagents.tracing import traced

    def sample(x: int) -> int:
        return x * 2

    wrapped = traced(sample)
    assert wrapped(21) == 42


def test_run_swarm_traced_offline(monkeypatch):
    class FakeModel:
        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages, config=None, **kwargs):
            from langchain_core.messages import AIMessage

            return AIMessage(content="fake answer")

    monkeypatch.setattr(
        "swarmagents.tracing.get_chat_model", lambda *a, **k: FakeModel()
    )
    monkeypatch.setattr("swarmagents.swarm.get_chat_model", lambda *a, **k: FakeModel())
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    from swarmagents.tracing import run_swarm_traced

    result = run_swarm_traced("hello", provider="openai")
    assert isinstance(result, str)


def test_graphview_topology():
    from swarmagents.graphview import swarm_topology

    topo = swarm_topology()
    names = {a["name"] for a in topo["agents"]}
    assert names == {"researcher", "analyst", "writer"}
    assert len(topo["edges"]) == 6


def test_graphview_cypher_and_mermaid(tmp_path):
    from swarmagents.graphview import export, swarm_topology, to_cypher, to_mermaid

    topo = swarm_topology()
    cypher = to_cypher(topo)
    assert 'MERGE (a:Agent {name: "researcher"})' in cypher
    assert "HANDS_OFF_TO" in cypher
    mermaid = to_mermaid(topo)
    assert mermaid.startswith("graph LR")
    assert "researcher -->|hands off| analyst" in mermaid

    result = export(path=str(tmp_path))
    assert result["neo4j_pushed"] is False
    assert (tmp_path / "swarm_graph.cypher").exists()
    assert (tmp_path / "swarm_graph.mmd").exists()


def test_specialist_roles_defined():
    from swarmagents.agents import SPECIALIST_ROLES

    expected = {
        "codereviewer",
        "debugger",
        "coder",
        "memory",
        "deepresearch",
        "guidecreator",
        "judge",
        "caching",
        "graphcreator",
        "compilator",
        "refactor",
        "optimizer",
        "filemanager",
    }
    assert set(SPECIALIST_ROLES) == expected
    for role, prompt in SPECIALIST_ROLES.items():
        assert len(prompt) > 60, f"prompt too thin for {role}"


def test_specialist_swarm_builds(monkeypatch):
    class FakeModel:
        def bind_tools(self, tools, **kwargs):
            return self

    monkeypatch.setattr(
        "swarmagents.agents.create_agent",
        lambda model, tools, system_prompt, name: {
            "model": model,
            "tools": tools,
            "name": name,
        },
    )
    monkeypatch.setattr(
        "swarmagents.agents.create_swarm",
        lambda agents, default_active_agent: {
            "agents": agents,
            "default": default_active_agent,
        },
    )
    from swarmagents.agents import create_specialist_swarm

    swarm = create_specialist_swarm(FakeModel())
    assert swarm["default"] == "codereviewer"
    assert len(swarm["agents"]) == 13
    from swarmagents.agents import SPECIALIST_ROLES

    names = {a["name"] for a in swarm["agents"]}
    assert names == set(SPECIALIST_ROLES)


def test_rust_cache_roundtrip_and_eviction():
    from swarmagents_core import Int8Cache

    c = Int8Cache(3)
    for i in range(5):
        c.put(f"k{i}", [i, i + 1])
    assert c.len() <= 3
    assert c.capacity() == 3
    c.put("hit", [9, 9])
    assert c.get("hit") == [9, 9]
    assert c.get("definitely-missing") is None


def test_fan_out_parallel(monkeypatch):
    from swarmagents import parallel as par

    class FakeModel:
        def bind_tools(self, tools, **kwargs):
            return self

    seen = []

    def fake_build(role, model, handoffs, tools):
        seen.append(role)

        class Agent:
            def invoke(self, payload, config=None):
                class Msg:
                    content = f"answer from {role}"

                return {"messages": [Msg()]}

        return Agent()

    monkeypatch.setattr(par, "build_specialist", fake_build)
    out = par.fan_out(("coder", "deepresearch", "judge"), "do the thing", FakeModel())
    assert set(out) == {"coder", "deepresearch", "judge"}
    assert out["coder"] == "answer from coder"
    assert len(seen) == 3


def test_fan_out_rejects_unknown_role():
    import pytest

    from swarmagents import parallel as par

    with pytest.raises(ValueError, match="unknown roles"):
        par.fan_out(("not-a-role",), "x", "openai")


def test_judge_over_candidates(monkeypatch):
    from swarmagents import parallel as par

    class FakeModel:
        def bind_tools(self, tools, **kwargs):
            return self

    calls = []

    def fake_build(role, model, handoffs, tools):
        calls.append(role)

        class Agent:
            def invoke(self, payload, config=None):
                content = payload["messages"][0]["content"]
                suffix = "verdict" if "Candidates:" in content else role
                return {"messages": [type("M", (), {"content": f"out:{suffix}"})()]}

        return Agent()

    monkeypatch.setattr(par, "build_specialist", fake_build)
    result = par.judge_over_candidates(
        ("coder", "optimizer"), "improve this", FakeModel()
    )
    assert set(result["candidates"]) == {"coder", "optimizer"}
    assert result["verdict"].startswith("out:")
    assert "judge" in calls


def test_parallel_backend_report():
    from swarmagents.parallel import parallel_backend

    report = parallel_backend()
    assert "threads" in report


def test_extract_facts_offline():
    from swarmagents.extraction import extract_facts

    facts = extract_facts("The deployment failed at 14:02 due to a timeout.")
    assert isinstance(facts, list)
    assert all(isinstance(f, dict) for f in facts)


def test_neo4j_retriever_none_without_credentials(monkeypatch):
    from swarmagents.extraction import neo4j_retriever

    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    assert neo4j_retriever() is None
