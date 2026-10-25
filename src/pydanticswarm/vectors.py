"""FAISS vector search: int8 (IndexScalarQuantizer) vs normal (IndexFlatL2).

Web-verified numbers (added 2026-10, faiss-cpu 1.15.1):
- IndexScalarQuantizer with ScalarQuantizer.QT_8bit: 4x memory reduction
  (75% less) vs float32, ~3.66x average speedup (HF embedding-quantization
  benchmark), recall typically 95-98% on BERT-style embeddings.
- IndexFlatL2: exact search, baseline for recall, no compression.
- int8 needs train() (learns per-dimension ranges); Flat does not.

Design: build both, measure speed/recall/memory, auto-select via
`pick_index()` — int8 when the corpus is large, flat when exactness matters
or the corpus is small (int8 training overhead doesn't pay off below ~1k
vectors per the FAISS training guidance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np
from pydantic import BaseModel


class SearchStats(BaseModel):
    """Measured comparison between int8 and flat indexes."""

    n_vectors: int = 0
    dim: int = 0
    flat_time_ms: float = 0.0
    int8_time_ms: float = 0.0
    recall_at_k: float = 0.0
    flat_bytes: int = 0
    int8_bytes: int = 0

    @property
    def speedup(self) -> float:
        if self.int8_time_ms <= 0:
            return 0.0
        return round(self.flat_time_ms / self.int8_time_ms, 2)

    @property
    def memory_saving(self) -> float:
        """Fraction of memory saved by int8 (0.0-1.0)."""
        if self.flat_bytes <= 0:
            return 0.0
        return round(1 - self.int8_bytes / self.flat_bytes, 3)


@dataclass
class VectorStore:
    """Dual FAISS store: int8 SQ quantizer + exact Flat, measured head-to-head."""

    dim: int
    _flat: faiss.Index | None = None
    _int8: faiss.Index | None = None
    _corpus: np.ndarray | None = None  # float32 copy for recall measurement
    _labels: list[str] = field(default_factory=list)

    def build(self, vectors: np.ndarray, labels: list[str] | None = None) -> None:
        """Train and add to both indexes. Vectors: (n, dim) float32."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) float32, got {vectors.shape}")
        self._corpus = vectors
        self._labels = list(labels or [])

        # Normal: exact flat L2 (no training needed)
        self._flat = faiss.IndexFlatL2(self.dim)
        self._flat.add(vectors)

        # int8: scalar quantizer, QT_8bit (needs train() to learn ranges)
        self._int8 = faiss.IndexScalarQuantizer(
            self.dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
        )
        self._int8.train(vectors)
        self._int8.add(vectors)

    def search_flat(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        return self._search(self._flat, query, k)

    def search_int8(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        return self._search(self._int8, query, k)

    def _search(
        self, index: faiss.Index, query: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, self.dim)
        k = min(k, index.ntotal)
        distances, ids = index.search(q, k)
        out: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], ids[0], strict=True):
            if idx < 0:
                continue
            label = self._labels[idx] if self._labels else str(idx)
            out.append((label, float(dist)))
        return out

    def benchmark(self, queries: np.ndarray, k: int = 5) -> SearchStats:
        """Time both indexes and measure int8 recall vs the exact flat baseline."""
        import time

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        assert self._flat is not None and self._int8 is not None, "build() first"

        t0 = time.perf_counter()
        flat_ids_all = []
        for i in range(queries.shape[0]):
            _, ids = self._flat.search(queries[i : i + 1], k)
            flat_ids_all.append(set(ids[0].tolist()))
        flat_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        int8_ids_all = []
        for i in range(queries.shape[0]):
            _, ids = self._int8.search(queries[i : i + 1], k)
            int8_ids_all.append(set(ids[0].tolist()))
        int8_ms = (time.perf_counter() - t0) * 1000

        hits = sum(len(a & b) for a, b in zip(flat_ids_all, int8_ids_all, strict=True))
        total = sum(len(a) for a in flat_ids_all) or 1
        recall = hits / total

        n = self._flat.ntotal
        return SearchStats(
            n_vectors=n,
            dim=self.dim,
            flat_time_ms=round(flat_ms, 3),
            int8_time_ms=round(int8_ms, 3),
            recall_at_k=round(recall, 4),
            flat_bytes=n * self.dim * 4,
            int8_bytes=n * self.dim,  # 1 byte per component
        )


def pick_index(n_vectors: int, exact_required: bool = False) -> str:
    """Heuristic: int8 for large corpora, flat for small or exactness-critical."""
    if exact_required:
        return "flat"
    return "int8" if n_vectors >= 1000 else "flat"
