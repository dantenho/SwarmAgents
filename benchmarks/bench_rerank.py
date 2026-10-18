"""Rerank top-k cosine benchmark: no-JIT vs free-threaded backends.

Reports avg ms per rerank call and the delta between backends. Run on a
free-threaded build for the parallel path:

    PYTHON_GIL=0 PYTHONPATH=. .venv-315t/bin/python benchmarks/bench_rerank.py
    PYTHONPATH=. .venv/bin/python benchmarks/bench_rerank.py
"""

from __future__ import annotations

import random
import statistics
import sys
import time

from swarmagents.kernels import active_backend, gil_enabled

DIM = 384
CORPUS = 2_000
K = 10
REPEAT = 20


def cosine(a: list[int], b: list[int]) -> float:
    dot = 0
    na = 0
    nb = 0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


def rerank_nojit(query, corpus, k):
    scores = [cosine(query, c) for c in corpus]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(i, scores[i]) for i in ranked[:k]]


def rerank_freethread(query, corpus, k, workers):
    from concurrent.futures import ThreadPoolExecutor

    def score_chunk(chunk):
        return [(i, cosine(query, c)) for i, c in chunk]

    indexed = list(enumerate(corpus))
    size = max(1, (len(indexed) + workers - 1) // workers)
    chunks = [indexed[i : i + size] for i in range(0, len(indexed), size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(score_chunk, chunks))
    flat = [pair for part in parts for pair in part]
    flat.sort(key=lambda p: p[1], reverse=True)
    return flat[:k]


def avg_ms(fn, *args, repeat: int = REPEAT) -> float:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(samples)


def main() -> None:
    print(f"python {sys.version.split()[0]}  GIL enabled: {gil_enabled()}")
    print(f"kernels backend: {active_backend()}")
    print(f"rerank: corpus={CORPUS} dim={DIM} k={K} repeat={REPEAT}")
    print("-" * 56)

    rng = random.Random(42)
    corpus = [[rng.randint(-127, 127) for _ in range(DIM)] for _ in range(CORPUS)]
    query = [rng.randint(-127, 127) for _ in range(DIM)]

    a = rerank_nojit(query, corpus, K)
    b = rerank_freethread(query, corpus, K, 8)
    assert a[0][0] == b[0][0], f"backend disagreement: {a[0]} vs {b[0]}"
    print(f"  top-1 agreement OK (doc {a[0][0]}, score {a[0][1]:.4f})")

    t_nojit = avg_ms(rerank_nojit, query, corpus, K)
    print(f"{'rerank no-jit (1 thread)':30s} avg {t_nojit:9.3f} ms")

    t_ft = avg_ms(rerank_freethread, query, corpus, K, 8)
    print(f"{'rerank freethread (8 workers)':30s} avg {t_ft:9.3f} ms")

    delta = t_nojit - t_ft
    speedup = t_nojit / t_ft if t_ft else 0.0
    print("-" * 56)
    print(f"  delta avg: {delta:+.3f} ms  speedup: {speedup:.2f}x")
    if gil_enabled():
        print("  note: GIL active — threaded path serializes; expect ~1x")
    else:
        print("  note: GIL disabled — threaded path parallelizes for real")


if __name__ == "__main__":
    main()
