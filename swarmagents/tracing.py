"""LangSmith tracing integration for the swarm.

Wraps the public swarm entrypoints with ``@traceable`` so every run is
recorded as a LangSmith run tree when tracing is enabled
(``LANGSMITH_TRACING=true`` plus ``LANGSMITH_API_KEY``). Without credentials
the decorators are inert and add negligible overhead.

Enable:
    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY=lsv2_...
    export LANGSMITH_PROJECT=swarmagents
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING, Any

from swarmagents.providers import get_chat_model

if TYPE_CHECKING:
    from langchain_core.tracers import BaseTracer


def tracing_enabled() -> bool:
    """Return True when LangSmith tracing is configured."""
    return os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes") and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def _have_langsmith() -> bool:
    return importlib.util.find_spec("langsmith") is not None


def get_tracer() -> "BaseTracer | None":
    """Build a LangChain tracer when tracing is enabled, else None.

    Returns:
        A LangChainTracer wired to LangSmith, or None when disabled.
    """
    if not (tracing_enabled() and _have_langsmith()):
        return None
    from langchain_core.tracers import LangChainTracer

    tracer = LangChainTracer(project_name=os.getenv("LANGSMITH_PROJECT", "swarmagents"))
    return tracer


def traced(fn: Any) -> Any:
    """Decorate fn with LangSmith tracing when available and enabled.

    Args:
        fn: The swarm entrypoint to trace.

    Returns:
        The (possibly) traceable-wrapped function; fn unchanged when
        langsmith is missing or tracing is disabled.
    """
    if not _have_langsmith():
        return fn
    from langsmith import traceable

    return traceable(name=fn.__name__, run_type="chain")(fn)


def run_swarm_traced(
    message: str,
    provider: str = "openai",
    model: str | None = None,
    **model_kwargs: Any,
) -> str:
    """Run the handoff swarm with LangSmith tracing when configured.

    Args:
        message: The user's request for the swarm.
        provider: LLM API provider name from the 20-provider registry.
        model: Model name override.
        **model_kwargs: Extra kwargs forwarded to the chat model.

    Returns:
        The final assistant message content.
    """
    from swarmagents.swarm import create_langgraph_swarm

    get_chat_model(provider, model, **model_kwargs)
    graph = create_langgraph_swarm(provider, model, **model_kwargs)
    if hasattr(graph, "compile") and not hasattr(graph, "invoke"):
        swarm = graph.compile()
    else:
        swarm = graph
    tracer = get_tracer()
    config: dict[str, Any] = {}
    if tracer is not None:
        config["callbacks"] = [tracer]
    result = swarm.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config or None,
    )
    return str(result["messages"][-1].content)


__all__ = ("get_tracer", "run_swarm_traced", "traced", "tracing_enabled")
