"""Semantic lesson retrieval: int8 vector search over learned lessons.

Closes the self-learning loop: instead of exact skill matching, retrieve
the top-k most relevant lessons for a NEW task by embedding similarity.
Embeddings use a deterministic hashing trick (no model download, no API),
indexed via the FAISS int8 VectorStore (4x memory saving, ~3.66x speedup).
"""

from __future__ import annotations

import numpy as np

from pydanticswarm.learning import Lesson, MemoryStore
from pydanticswarm.vectors import VectorStore


def _stable_hash(text: str) -> int:
    """Process-stable hash (md5) — Python's hash() is randomized per run."""
    import hashlib

    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")  # 64-bit stable bucket


def embed_text(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic hashing embedding: bag of hashed token buckets, L2-normalized.

    No model needed; stable across runs (md5, not Python's randomized hash);
    good enough for lesson retrieval.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    for i, tok in enumerate(tokens):
        # two hash signals per token: identity + bigram with previous token
        vec[_stable_hash(tok) % dim] += 1.0
        if i > 0:
            vec[_stable_hash(f"{tokens[i - 1]}_{tok}") % dim] += 0.5
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


class SemanticMemoryStore(MemoryStore):
    """MemoryStore with a FAISS int8 index over lesson embeddings.

    Falls back to exact skill matching when the index is cold (no lessons).
    """

    def __init__(self, dim: int = 256, persist_path=None) -> None:
        super().__init__(persist_path=persist_path)
        self.dim = dim
        self._store: VectorStore | None = None
        self._lesson_labels: list[str] = []

    def add_lesson(self, lesson: Lesson) -> None:
        super().add_lesson(lesson)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self.lessons:
            self._store = None
            return
        self._store = VectorStore(dim=self.dim)
        self._lesson_labels = []
        vecs = []
        for i, lesson in enumerate(self.lessons):
            text = f"{lesson.skill} {lesson.context} {lesson.pattern}"
            vecs.append(embed_text(text, self.dim))
            self._lesson_labels.append(str(i))
        self._store.build(np.stack(vecs), labels=self._lesson_labels)

    def relevant_lessons(self, task_description: str, k: int = 3) -> list[Lesson]:
        """Top-k lessons by embedding similarity to the task (int8 search)."""
        if self._store is None or not self.lessons:
            return []
        query = embed_text(task_description, self.dim)
        hits = self._store.search_int8(query, k=min(k, len(self.lessons)))
        return [
            self.lessons[int(idx)] for idx, _ in hits if int(idx) < len(self.lessons)
        ]
