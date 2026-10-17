"""Registry of 20 LLM API providers with lazy model construction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


class ProviderError(RuntimeError):
    """Raised when a provider cannot be resolved or instantiated."""


@dataclass(frozen=True)
class ProviderSpec:
    """Description of one LLM API provider."""

    name: str
    default_model: str
    api_key_env: str | None
    integration: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None

    @property
    def openai_compatible(self) -> bool:
        return self.integration is None and self.base_url is not None


REGISTRY: dict[str, ProviderSpec] = {
    spec.name: spec
    for spec in (
        ProviderSpec(
            name="openai",
            default_model="gpt-4.1",
            api_key_env="OPENAI_API_KEY",
            integration="openai",
        ),
        ProviderSpec(
            name="anthropic",
            default_model="claude-sonnet-4-5",
            api_key_env="ANTHROPIC_API_KEY",
            integration="anthropic",
        ),
        ProviderSpec(
            name="google-genai",
            default_model="gemini-2.5-pro",
            api_key_env="GOOGLE_API_KEY",
            integration="google_genai",
        ),
        ProviderSpec(
            name="bedrock",
            default_model="anthropic.claude-sonnet-4-5",
            api_key_env="AWS_ACCESS_KEY_ID",
            integration="bedrock",
        ),
        ProviderSpec(
            name="mistralai",
            default_model="mistral-large-latest",
            api_key_env="MISTRAL_API_KEY",
            integration="mistralai",
        ),
        ProviderSpec(
            name="groq",
            default_model="llama-3.3-70b-versatile",
            api_key_env="GROQ_API_KEY",
            integration="groq",
        ),
        ProviderSpec(
            name="fireworks",
            default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            api_key_env="FIREWORKS_API_KEY",
            integration="fireworks",
        ),
        ProviderSpec(
            name="together",
            default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key_env="TOGETHER_API_KEY",
            integration="together",
        ),
        ProviderSpec(
            name="deepseek",
            default_model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
            integration="deepseek",
        ),
        ProviderSpec(
            name="xai",
            default_model="grok-4",
            api_key_env="XAI_API_KEY",
            integration="xai",
        ),
        ProviderSpec(
            name="cohere",
            default_model="command-r-plus",
            api_key_env="COHERE_API_KEY",
            integration="cohere",
        ),
        ProviderSpec(
            name="perplexity",
            default_model="sonar-pro",
            api_key_env="PERPLEXITY_API_KEY",
            integration="perplexity",
        ),
        ProviderSpec(
            name="huggingface",
            default_model="meta-llama/Llama-3.3-70B-Instruct",
            api_key_env="HF_TOKEN",
            integration="huggingface",
        ),
        ProviderSpec(
            name="ollama",
            default_model="llama3.1",
            api_key_env=None,
            integration="ollama",
        ),
        ProviderSpec(
            name="vllm",
            default_model="Qwen/Qwen2.5-72B-Instruct",
            api_key_env=None,
            base_url="http://localhost:8000/v1",
            base_url_env="VLLM_BASE_URL",
        ),
        ProviderSpec(
            name="openrouter",
            default_model="openai/gpt-4.1",
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
        ),
        ProviderSpec(
            name="novita",
            default_model="meta-llama/llama-3.3-70b-instruct",
            api_key_env="NOVITA_API_KEY",
            base_url="https://api.novita.ai/v3/openai",
        ),
        ProviderSpec(
            name="nvidia",
            default_model="meta/llama-3.3-70b-instruct",
            api_key_env="NVIDIA_API_KEY",
            base_url="https://integrate.api.nvidia.com/v1",
        ),
        ProviderSpec(
            name="cerebras",
            default_model="llama-3.3-70b",
            api_key_env="CEREBRAS_API_KEY",
            base_url="https://api.cerebras.ai/v1",
        ),
        ProviderSpec(
            name="azure-openai",
            default_model="gpt-4.1",
            api_key_env="AZURE_OPENAI_API_KEY",
            base_url=None,
            base_url_env="AZURE_OPENAI_ENDPOINT",
        ),
    )
}

PROVIDER_NAMES: tuple[str, ...] = tuple(REGISTRY)


def get_provider_spec(name: str) -> ProviderSpec:
    """Return the spec for a provider name (case-insensitive)."""
    key = name.lower()
    if key not in REGISTRY:
        known = ", ".join(PROVIDER_NAMES)
        raise ProviderError(
            f"Unknown provider {name!r}. Known providers: {known}"
        )
    return REGISTRY[key]


def _api_key(spec: ProviderSpec) -> str | None:
    if spec.api_key_env is None:
        return None
    key = os.getenv(spec.api_key_env)
    if not key:
        raise ProviderError(
            f"Provider {spec.name!r} requires an API key in "
            f"{spec.api_key_env}"
        )
    return key


def _integration_model(spec: ProviderSpec, model: str, **kwargs: Any) -> BaseChatModel:
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as err:
        raise ProviderError(
            "The 'langchain' package is required to load provider models"
        ) from err
    return init_chat_model(  # type: ignore[no-any-return]
        model=model,
        model_provider=spec.integration,
        api_key=_api_key(spec),
        **kwargs,
    )


def _openai_compatible_model(
    spec: ProviderSpec, model: str, **kwargs: Any
) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as err:
        raise ProviderError(
            "Install 'langchain-openai' to use the "
            f"{spec.name!r} provider: pip install langchain-openai"
        ) from err
    return ChatOpenAI(  # type: ignore[no-any-return]
        model=model,
        api_key=_api_key(spec),
        base_url=spec.base_url or os.getenv(spec.base_url_env or "", None),
        **kwargs,
    )


def _azure_model(spec: ProviderSpec, model: str, **kwargs: Any) -> BaseChatModel:
    endpoint = os.getenv(spec.base_url_env or "", "")
    if not endpoint:
        raise ProviderError(
            f"Provider {spec.name!r} requires an endpoint in "
            f"{spec.base_url_env}"
        )
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError as err:
        raise ProviderError(
            "Install 'langchain-openai' to use the "
            f"{spec.name!r} provider: pip install langchain-openai"
        ) from err
    return AzureChatOpenAI(  # type: ignore[no-any-return]
        azure_endpoint=endpoint,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        model=model,
        api_key=_api_key(spec),
        **kwargs,
    )


def get_chat_model(
    provider: str, model: str | None = None, **kwargs: Any
) -> BaseChatModel:
    """Build a chat model instance for one of the 20 providers."""
    spec = get_provider_spec(provider)
    resolved_model = model or spec.default_model
    if spec.name == "azure-openai":
        return _azure_model(spec, resolved_model, **kwargs)
    if spec.integration is not None:
        return _integration_model(spec, resolved_model, **kwargs)
    return _openai_compatible_model(spec, resolved_model, **kwargs)
