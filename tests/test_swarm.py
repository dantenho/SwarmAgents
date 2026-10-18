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
