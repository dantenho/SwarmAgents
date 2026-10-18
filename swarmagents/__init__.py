"""SwarmAgents: LangChain/LangGraph agentic swarm workflows.

All heavy submodule imports are deferred (PEP 810 lazy imports) so the
package imports cleanly on interpreters without langchain/numba installed
(e.g. the 3.15 free-threaded venv).
"""

from swarmagents.embeddings import EMBEDDING_DIM, MODEL_NAME, embed, embed_batch
from swarmagents.kernels import (
    active_backend,
    batch_cosine,
    cosine_int8,
    top_k,
)
from swarmagents.agents import (
    DEFAULT_SPECIALISTS,
    SPECIALIST_ROLES,
    create_specialist_swarm,
)
from swarmagents.vectordb import (
    LocalVectorStore,
    SupabaseVectorStore,
    create_vector_store,
)

__all__ = (
    "EMBEDDING_DIM",
    "MODEL_NAME",
    "active_backend",
    "batch_cosine",
    "cosine_int8",
    "embed",
    "embed_batch",
    "get_chat_model",
    "get_provider_spec",
    "run_langgraph_swarm",
    "run_swarm",
    "top_k",
    "LocalVectorStore",
    "SupabaseVectorStore",
    "create_vector_store",
    "DEFAULT_SPECIALISTS",
    "SPECIALIST_ROLES",
    "create_specialist_swarm",
)
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "PROVIDER_NAMES":
        from swarmagents.providers import PROVIDER_NAMES

        return PROVIDER_NAMES
    if name in ("get_chat_model", "get_provider_spec"):
        from swarmagents import providers

        return getattr(providers, name)
    if name == "run_langgraph_swarm":
        from swarmagents.swarm import run_langgraph_swarm

        return run_langgraph_swarm
    if name == "create_langgraph_swarm":
        from swarmagents.swarm import create_langgraph_swarm

        return create_langgraph_swarm
    if name == "AGENT_ROLES":
        from swarmagents.swarm import AGENT_ROLES

        return AGENT_ROLES
    if name in ("run_swarm",):
        from swarmagents.graph import run_swarm

        return run_swarm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
