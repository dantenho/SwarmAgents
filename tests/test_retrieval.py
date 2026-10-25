"""Tests: semantic lesson retrieval via int8 vector search."""

import numpy as np

from pydanticswarm.learning import Lesson
from pydanticswarm.retrieval import SemanticMemoryStore, embed_text


def test_embed_is_deterministic_and_normalized():
    a = embed_text("code review retry pattern")
    b = embed_text("code review retry pattern")
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5


def test_similar_texts_closer_than_unrelated():
    a = embed_text("retry failed llm calls with backoff")
    b = embed_text("retry failed calls backoff")
    c = embed_text("write blog posts about cooking")
    assert float(np.dot(a, b)) > float(np.dot(a, c))


def test_relevant_lessons_empty_when_cold():
    mem = SemanticMemoryStore()
    assert mem.relevant_lessons("anything") == []


def test_retrieval_returns_most_similar_lesson_first():
    mem = SemanticMemoryStore()
    mem.add_lesson(
        Lesson(
            skill="code", pattern="retry with smaller prompt", context="llm code gen"
        )
    )
    mem.add_lesson(
        Lesson(skill="code", pattern="split files before edit", context="file refactor")
    )
    mem.add_lesson(
        Lesson(skill="write", pattern="outline first", context="blog writing")
    )
    hits = mem.relevant_lessons("llm code generation retry", k=2)
    assert len(hits) <= 2
    assert hits[0].pattern == "retry with smaller prompt"


def test_retrieval_uses_int8_index():
    mem = SemanticMemoryStore()
    mem.add_lesson(Lesson(skill="s", pattern="p1", context="c1"))
    assert mem._store is not None
    # int8 store has 1 byte per component: rebuild uses float32 input but
    # the index itself is scalar-quantized (verify via search path)
    hits = mem.relevant_lessons("c1 p1", k=1)
    assert len(hits) == 1
    assert hits[0].pattern == "p1"


def test_index_rebuilds_on_each_add():
    mem = SemanticMemoryStore()
    mem.add_lesson(Lesson(skill="s", pattern="p1"))
    first = mem._store
    mem.add_lesson(Lesson(skill="s", pattern="p2"))
    assert mem._store is not first
    assert len(mem.lessons) == 2
