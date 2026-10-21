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
from pydanticswarm.orchestrator import Orchestrator
from pydanticswarm.swarm import AgentResult, AgentSpec, Subtask, Swarm, SwarmOutcome

__all__ = [
    "AffinityTracker",
    "AgentResult",
    "AgentSpec",
    "Blackboard",
    "DivisionMode",
    "DivisionOutcome",
    "Envelope",
    "Lesson",
    "MemoryStore",
    "MessageBus",
    "MessageKind",
    "Orchestrator",
    "Outcome",
    "Subtask",
    "Swarm",
    "SwarmOutcome",
    "divide_different",
    "divide_multiple",
    "divide_same",
]
__version__ = "0.2.0"
