"""LangChain full benchmark and metrics suite.

Measures the LangSwarm stack end to end without API keys:
- provider registry resolution
- embedding-cache hit/miss latency (fake vectors)
- kernels backend dispatch + batch/top-k throughput
- LangGraph swarm construction cost (fake model)
- Rust pyo3 vs Python similarity

Run: PYTHONPATH=. .venv/bin/python benchmarks/bench_langchain.py
"""

from __future__ import annotations

import statistics
import time
from typing import Any

DIM = 384


def timed(label: str, fn, *args, repeat: int = 5) -> dict[str, Any]:
    """Run fn repeat times, return a metrics record."""
    samples = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        samples.append((time.perf_counter() - t0) * 1000)
    record = {
        "label": label,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "result": result,
    }
    print(
        f"{label:34s} median {record['median_ms']:9.3f} ms  "
        f"(min {record['min_ms']:.3f} max {record['max_ms']:.3f} "
        f"sd {record['stdev_ms']:.3f})"
    )
    return record


def bench_provider_registry() -> dict[str, Any]:
    from swarmagents.providers import PROVIDER_NAMES, get_provider_spec

    def resolve_all():
        return [get_provider_spec(n) for n in PROVIDER_NAMES]

    record = timed("provider registry resolve (20)", resolve_all)
    assert isinstance(record["result"], list) and len(record["result"]) == 20
    return record


def bench_embeddings_cache() -> dict[str, Any]:
    import hashlib
    import json

    class FakeRedis:
        def __init__(self):
            self.store = {}
            self.hits = 0
            self.misses = 0

        def get(self, key):
            if key in self.store:
                self.hits += 1
                return self.store[key]
            self.misses += 1
            return None

        def set(self, key, value, ex=None):
            self.store[key] = value

    fake = FakeRedis()

    def fake_encode(text):
        digest = hashlib.sha256(text.encode()).digest()
        return [b - 128 for b in digest[: DIM // 3]] * 3

    def embed_cached(text):
        key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
        cached = fake.get(key)
        if cached is not None:
            return json.loads(cached)
        vector = fake_encode(text)
        fake.set(key, json.dumps(vector), ex=3600)
        return vector

    text = "benchmark text " * 4
    embed_cached(text)
    fake.hits = 0
    fake.misses = 0
    record = timed("embed cache hit (fake store)", embed_cached, text, repeat=20)
    return {"record": record, "hits": fake.hits, "misses": fake.misses}


def bench_kernels() -> dict[str, Any]:
    import random

    from swarmagents.kernels import active_backend, batch_cosine, top_k

    rng = random.Random(42)
    corpus = [[rng.randint(-127, 127) for _ in range(DIM)] for _ in range(1000)]
    query = corpus[0]

    print(f"kernels backend: {active_backend()}")
    r_batch = timed("batch_cosine 1k vectors", batch_cosine, corpus, corpus)
    r_topk = timed("top_k over 1k corpus", top_k, query, corpus, 5)
    best = r_topk["result"][0]
    assert best[0] == 0 and best[1] == 1.0, f"unexpected top1: {best}"
    throughput = 1_000_000 / r_batch["median_ms"] * 2
    print(f"  ~{throughput:,.0f} comparisons/sec (2x1000 per batch call)")
    return {"batch": r_batch, "top_k": r_topk}


def bench_swarm_construction() -> dict[str, Any]:
    class FakeModel:
        def bind_tools(self, tools, **kwargs):
            return self

    import swarmagents.swarm as swarm_mod
    from swarmagents import providers as providers_mod

    orig = providers_mod.get_chat_model
    orig_swarm = swarm_mod.get_chat_model
    providers_mod.get_chat_model = lambda *a, **k: FakeModel()  # type: ignore[assignment]
    try:
        swarm_mod.get_chat_model = lambda *a, **k: FakeModel()  # type: ignore[assignment]
        build = swarm_mod.create_langgraph_swarm
        record = timed("build handoff swarm (fake model)", build, "openai")
        built = record["result"]
        assert built is not None
    finally:
        providers_mod.get_chat_model = orig  # type: ignore[assignment]
        swarm_mod.get_chat_model = orig_swarm  # type: ignore[assignment]
    return record


def bench_rust_vs_python() -> dict[str, Any] | None:
    import random

    try:
        from swarmagents_core import (  # type: ignore[attr-defined]
            cosine_similarity_int8 as rust_cosine,
        )
    except ImportError:
        print("pyo3/rust: not installed, skipping")
        return None

    from swarmagents.kernels import cosine_int8 as py_cosine

    rng = random.Random(42)
    a = [rng.randint(-127, 127) for _ in range(DIM)]
    b = [rng.randint(-127, 127) for _ in range(DIM)]
    assert abs(rust_cosine(a, b) - py_cosine(a, b)) < 1e-9

    r_rust = timed("rust cosine (single)", rust_cosine, a, b, repeat=20)
    r_py = timed("python cosine (single)", py_cosine, a, b, repeat=20)
    speedup = r_py["median_ms"] / r_rust["median_ms"]
    print(f"  rust speedup: {speedup:.1f}x")
    return {"rust": r_rust, "python": r_py, "speedup": speedup}


def main() -> None:
    print("LangSwarm full benchmark suite")
    print("=" * 60)
    metrics: dict[str, Any] = {}

    metrics["registry"] = bench_provider_registry()
    print()
    metrics["embeddings"] = bench_embeddings_cache()
    print(
        f"  cache hits={metrics['embeddings']['hits']} "
        f"misses={metrics['embeddings']['misses']}"
    )
    print()
    metrics["kernels"] = bench_kernels()
    print()
    metrics["swarm"] = bench_swarm_construction()
    print()
    metrics["rust"] = bench_rust_vs_python()

    print()
    print("summary")
    print("-" * 60)
    for key, value in metrics.items():
        if key == "embeddings":
            rec = value["record"]
            print(f"  {key:12s} median {rec['median_ms']:.3f} ms")
        elif key == "rust" and value:
            print(f"  {key:12s} speedup {value['speedup']:.1f}x")
        else:
            rec = value
            if isinstance(rec, dict) and "median_ms" in rec:
                print(f"  {key:12s} median {rec['median_ms']:.3f} ms")


if __name__ == "__main__":
    main()
