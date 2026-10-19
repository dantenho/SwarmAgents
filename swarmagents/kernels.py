"""Adaptive compute kernels for int8 similarity with best-backend dispatch.

Selection order at call time (all imports are lazy, PEP 810 style):
1. numba ``@njit(parallel=True)`` when importable (CPython <=3.14)
2. free-threaded ThreadPool when the GIL is disabled (3.14t/3.15t)
3. pyo3 ``swarmagents_core`` when the Rust extension is installed
4. pure Python fallback
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

DIM = 384
_MAX_WORKERS = min(32, (os.cpu_count() or 4) * 4)

_numba_batch = None
_numba_attempted = False


def gil_enabled() -> bool:
    """Return True when the GIL is active (or unknown)."""
    try:
        return bool(sys._is_gil_enabled())
    except AttributeError:
        return True


def _have_numba() -> bool:
    return importlib.util.find_spec("numba") is not None


def _have_rust() -> bool:
    return importlib.util.find_spec("swarmagents_core") is not None


def _load_numba():
    global _numba_batch, _numba_attempted
    if _numba_attempted:
        return _numba_batch
    _numba_attempted = True
    if not _have_numba():
        return None
    try:
        import numpy as np
        from numba import njit, prange

        @njit(cache=True, parallel=True)
        def _batch(a, b):
            out = np.empty(a.shape[0])
            for i in prange(a.shape[0]):
                dot = 0
                na = 0
                nb = 0
                for j in range(a.shape[1]):
                    dot += a[i, j] * b[i, j]
                    na += a[i, j] * a[i, j]
                    nb += b[i, j] * b[i, j]
                denom = math.sqrt(na) * math.sqrt(nb)
                out[i] = dot / denom if denom else 0.0
            return out

        _batch(
            np.zeros((1, DIM), dtype=np.int8),
            np.zeros((1, DIM), dtype=np.int8),
        )
        _numba_batch = _batch
    except Exception:
        _numba_batch = None
    return _numba_batch


def cosine_int8(a: list[int], b: list[int]) -> float:
    """Pure-Python int8 cosine similarity."""
    dot = 0
    na = 0
    nb = 0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.isqrt(na * nb)
    if denom == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _cosine_chunk(pairs: list[tuple[list[int], list[int]]]) -> list[float]:
    return [cosine_int8(a, b) for a, b in pairs]


def batch_cosine(vectors_a: list[list[int]], vectors_b: list[list[int]]) -> list[float]:
    """Cosine similarity for aligned batches, dispatched to the best backend."""
    if len(vectors_a) != len(vectors_b):
        raise ValueError("batches must be aligned")
    if not vectors_a:
        return []

    batch = _load_numba()
    if batch is not None:
        import numpy as np

        a = np.asarray(vectors_a, dtype=np.int8)
        b = np.asarray(vectors_b, dtype=np.int8)
        return [float(x) for x in batch(a, b)]

    if not gil_enabled():
        pairs = list(zip(vectors_a, vectors_b, strict=True))
        chunk = max(1, len(pairs) // _MAX_WORKERS + 1)
        chunks = [pairs[i : i + chunk] for i in range(0, len(pairs), chunk)]
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            results = list(pool.map(_cosine_chunk, chunks))
        return [x for part in results for x in part]

    if _have_rust():
        from swarmagents_core import cosine_similarity_int8  # type: ignore[attr-defined]

        return [
            cosine_similarity_int8(a, b)
            for a, b in zip(vectors_a, vectors_b, strict=True)
        ]

    return [cosine_int8(a, b) for a, b in zip(vectors_a, vectors_b, strict=True)]


def top_k(
    query: list[int], corpus: list[list[int]], k: int = 5
) -> list[tuple[int, float]]:
    """Return the k most similar corpus entries as (index, score) pairs."""
    scores = batch_cosine([query] * len(corpus), corpus)
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


def active_backend() -> str:
    """Report which backend batch_cosine would currently use."""
    if _load_numba() is not None:
        return "numba-njit-parallel"
    if not gil_enabled():
        return f"freethread-pool-{_MAX_WORKERS}"
    if _have_rust():
        return "pyo3-rust"
    return "pure-python"
