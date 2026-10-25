"""PydanticSwarm — agent orchestration with plain Pydantic + asyncio."""

from pydanticswarm.affinity import AffinityTracker
from pydanticswarm.communication import (
    Blackboard,
    Envelope,
    MessageBus,
    MessageKind,
)
from pydanticswarm.division import (
    DivisionMode,
    DivisionOutcome,
    divide_different,
    divide_multiple,
    divide_same,
)
from pydanticswarm.learning import Lesson, MemoryStore, Outcome
from pydanticswarm.merge import (
    GoogleGenAIBackend,
    LLMBackend,
    MergedAgent,
    MergedPlan,
    MergedSwarm,
    PydanticAIBackend,
)
from pydanticswarm.orchestrator import Orchestrator
from pydanticswarm.parallel import PoolStats, WorkerPool, bounded_gather
from pydanticswarm.retrieval import SemanticMemoryStore, embed_text
from pydanticswarm.swarm import AgentResult, AgentSpec, Subtask, Swarm, SwarmOutcome
from pydanticswarm.tokens import PromptSpec, TokenMeter, TokenUsage, estimate_tokens
from pydanticswarm.vectors import SearchStats, VectorStore, pick_index

__all__ = [
    "AffinityTracker",
    "AgentResult",
    "AgentSpec",
    "Blackboard",
    "DivisionMode",
    "DivisionOutcome",
    "Envelope",
    "Lesson",
    "LLMBackend",
    "MemoryStore",
    "MergedAgent",
    "MergedPlan",
    "MergedSwarm",
    "MessageBus",
    "MessageKind",
    "PydanticAIBackend",
    "GoogleGenAIBackend",
    "Orchestrator",
    "SearchStats",
    "Outcome",
    "PoolStats",
    "PromptSpec",
    "SemanticMemoryStore",
    "Subtask",
    "Swarm",
    "SwarmOutcome",
    "TokenMeter",
    "TokenUsage",
    "WorkerPool",
    "bounded_gather",
    "divide_different",
    "divide_multiple",
    "divide_same",
    "estimate_tokens",
    "embed_text",
    "pick_index",
    "VectorStore",
]
__version__ = "0.3.0"
