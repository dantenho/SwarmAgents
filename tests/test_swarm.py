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
