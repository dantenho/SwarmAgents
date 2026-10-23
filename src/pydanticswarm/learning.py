"""Self-learning: outcome memory, success stats, and reusable lessons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class Outcome(BaseModel):
    """One completed (or failed) subtask by one agent."""

    agent_name: str
    skill: str
    subtask_id: str
    success: bool
    payload: Any = None
    error: str | None = None


class Lesson(BaseModel):
    """A reusable pattern extracted from a retry-that-succeeded."""

    skill: str
    pattern: str
    context: str = ""


class MemoryStore:
    """Fast loop (stats) + slow loop (lessons), JSON-persistable."""

    def __init__(self, persist_path: Path | None = None) -> None:
        self.outcomes: list[Outcome] = []
        self.lessons: list[Lesson] = []
        self.persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    def record(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)
        if self.persist_path is not None:
            self._save()

    def success_rate(self, agent_name: str, skill: str, default: float = 0.5) -> float:
        rows = [
            o for o in self.outcomes if o.agent_name == agent_name and o.skill == skill
        ]
        if not rows:
            return default
        return sum(1 for o in rows if o.success) / len(rows)

    def add_lesson(self, lesson: Lesson) -> None:
        self.lessons.append(lesson)
        if self.persist_path is not None:
            self._save()

    def lessons_for(self, skill: str) -> list[Lesson]:
        return [x for x in self.lessons if x.skill == skill]

    def _save(self) -> None:
        data = {
            "outcomes": [o.model_dump() for o in self.outcomes],
            "lessons": [x.model_dump() for x in self.lessons],
        }
        self.persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        data = json.loads(self.persist_path.read_text())
        self.outcomes = [Outcome(**o) for o in data.get("outcomes", [])]
        self.lessons = [Lesson(**x) for x in data.get("lessons", [])]
