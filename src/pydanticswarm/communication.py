"""Typed message bus + blackboard for agent-to-agent communication."""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageKind(str, Enum):
    REQUEST = "request"
    RESULT = "result"
    BROADCAST = "broadcast"
    ERROR = "error"
    HANDOFF = "handoff"


class Envelope(BaseModel):
    """Every agent-to-agent interaction is a validated envelope."""

    kind: MessageKind
    sender: str
    recipient: str | None = None
    task_id: str | None = None
    payload: Any = None
    ts: float = Field(default_factory=time.time)


class MessageBus:
    """In-process async bus. Agents never call each other directly."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Envelope]] = {}
        self.log: list[Envelope] = []

    def register(self, agent_name: str) -> asyncio.Queue[Envelope]:
        q: asyncio.Queue[Envelope] = asyncio.Queue()
        self._queues[agent_name] = q
        return q

    async def send(self, envelope: Envelope) -> None:
        self.log.append(envelope)
        if envelope.recipient is None:
            for q in self._queues.values():
                q.put_nowait(envelope)
            return
        q = self._queues.get(envelope.recipient)
        if q is not None:
            q.put_nowait(envelope)

    async def receive(self, agent_name: str, timeout: float = 5.0) -> Envelope | None:
        q = self._queues.get(agent_name)
        if q is None:
            return None
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except TimeoutError:
            return None


class Blackboard(BaseModel):
    """Shared state with an append-only history for replay/debug."""

    state: dict[str, Any] = Field(default_factory=dict)
    history: list[tuple[str, Any, float]] = Field(default_factory=list)

    def write(self, key: str, value: Any) -> None:
        self.state[key] = value
        self.history.append((key, value, time.time()))

    def read(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)
