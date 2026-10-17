"""SwarmAgents: LangChain/LangGraph agentic swarm workflows."""

from swarmagents.embeddings import EMBEDDING_DIM, MODEL_NAME, embed, embed_batch
from swarmagents.graph import run_swarm
from swarmagents.providers import PROVIDER_NAMES, get_chat_model, get_provider_spec
from swarmagents.swarm import (
    AGENT_ROLES,
    create_langgraph_swarm,
    run_langgraph_swarm,
)

__all__ = (
    "AGENT_ROLES",
    "EMBEDDING_DIM",
    "MODEL_NAME",
    "PROVIDER_NAMES",
    "create_langgraph_swarm",
    "embed",
    "embed_batch",
    "get_chat_model",
    "get_provider_spec",
    "run_langgraph_swarm",
    "run_swarm",
)
__version__ = "0.1.0"
