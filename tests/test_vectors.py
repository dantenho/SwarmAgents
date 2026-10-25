"""Tests: FAISS int8 vs flat (normal) vector search."""

import numpy as np
import pytest

from pydanticswarm.vectors import VectorStore, pick_index


def _make_data(n: int = 500, dim: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim)).astype(np.float32)


def test_build_rejects_bad_shape():
    store = VectorStore(dim=8)
    with pytest.raises(ValueError, match="expected"):
        store.build(np.zeros((3, 7), dtype=np.float32))


def test_flat_and_int8_return_same_top1_for_clean_queries():
    data = _make_data(n=200)
    store = VectorStore(dim=data.shape[1])
    store.build(data)
    # query with the exact vector of row 10 -> both must rank it first
    for q in [data[10], data[0], data[199]]:
        flat = store.search_flat(q, k=1)
        int8 = store.search_int8(q, k=1)
        assert flat[0][0] == int8[0][0]


def test_labels_round_trip():
    data = _make_data(n=10)
    labels = [f"doc-{i}" for i in range(10)]
    store = VectorStore(dim=data.shape[1])
    store.build(data, labels=labels)
    hits = store.search_flat(data[3], k=1)
    assert hits[0][0] == "doc-3"


def test_int8_recall_is_high_and_measured():
    """int8 SQ typically retains 95%+ recall vs exact flat."""
    data = _make_data(n=2000)
    queries = _make_data(n=50, seed=7)
    store = VectorStore(dim=data.shape[1])
    store.build(data)
    stats = store.benchmark(queries, k=5)
    assert stats.n_vectors == 2000
    assert stats.recall_at_k >= 0.90  # generous bound; typically ~1.0
    assert stats.memory_saving >= 0.70  # ~4x less memory


def test_benchmark_reports_speedup_and_memory():
    data = _make_data(n=1000)
    store = VectorStore(dim=data.shape[1])
    store.build(data)
    stats = store.benchmark(data[:5], k=3)
    assert stats.speedup >= 0  # timing varies; structure must be valid
    assert stats.flat_bytes == 1000 * 64 * 4
    assert stats.int8_bytes == 1000 * 64
    assert 0.0 <= stats.memory_saving <= 1.0


def test_pick_index_heuristic():
    assert pick_index(50) == "flat"
    assert pick_index(50, exact_required=True) == "flat"
    assert pick_index(5000) == "int8"
    assert pick_index(5000, exact_required=True) == "flat"
