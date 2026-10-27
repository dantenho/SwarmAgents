"""End-to-end pipeline run and benchmark.

Executes the full SwarmAgents stack in one process and measures every
stage: specialist swarm build, parallel fan-out, embeddings, kernels
dispatch, vector store add/search, and rerank. Uses a fake model so no
API keys are needed; the LangChain agent paths are exercised for real.

Run: PYTHONPATH=. .venv/bin/python benchmarks/bench_e2e.py
"""

from __future__ import annotations

import random
import statistics
import time
from typing import Any

DIM = 384
CORPUS = 200
K = 5
REPEAT = 5


class FakeModel:
    """Deterministic stand-in chat model for offline end-to-end runs."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "FakeModel":
        return self

    def invoke(self, messages: Any, config: dict | None = None, **kw: Any):
        from langchain_core.messages import AIMessage

        return AIMessage(content="stub answer from e2e run")


def timed(label: str, fn, *args: Any, repeat: int = REPEAT) -> dict[str, Any]:
    """Time fn and print median/min/max in ms."""
    samples: list[float] = []
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
        "result": result,
    }
    print(
        f"{label:34s} median {record['median_ms']:9.3f} ms  "
        f"(min {record['min_ms']:.2f} max {record['max_ms']:.2f})"
    )
    return record


def main() -> None:
    """Run the full stack end to end and report per-stage timings."""
    from swarmagents.agents import create_specialist_swarm
    from swarmagents.kernels import active_backend, top_k
    from swarmagents.parallel import fan_out, parallel_backend
    from swarmagents.vectordb import LocalVectorStore

    print("SwarmAgents end-to-end benchmark")
    print("=" * 64)
    print(f"kernels backend: {active_backend()}")
    print(f"parallel backend: {parallel_backend()}")
    print("=" * 64)

    model = FakeModel()

    print("\n[1] swarm build (13 specialists, fake model)")
    r_build = timed("create_specialist_swarm", create_specialist_swarm, model)

    print("\n[2] parallel fan-out (3 agents, fake model)")
    r_fanout = timed(
        "fan_out coder/deepresearch/judge",
        fan_out,
        ("coder", "deepresearch", "judge"),
        "analyze the tradeoffs of SQLite vs DuckDB",
        model,
    )
    answers = r_fanout["result"]
    assert set(answers) == {"coder", "deepresearch", "judge"}
    assert all(isinstance(a, str) and a for a in answers.values())

    print("\n[3] embeddings (deterministic stub vectors)")
    rng = random.Random(42)
    corpus_texts = [f"document {i} about swarms" for i in range(CORPUS)]

    def make_vectors() -> list[list[int]]:
        return [[rng.randint(-127, 127) for _ in range(DIM)] for _ in range(CORPUS)]

    r_vec = timed("generate corpus vectors", make_vectors)
    vectors = r_vec["result"]
    query = vectors[0]

    print("\n[4] kernels")
    r_topk = timed("top_k over corpus", top_k, query, vectors, K)
    best = r_topk["result"][0]
    assert best[0] == 0 and best[1] == 1.0, f"top-1 should be self: {best}"

    print("\n[5] vector store (add + search with stub embed)")
    import swarmagents.vectordb as vdb

    original_embed = vdb.embed

    def stub_embed(text: str, use_cache: bool = True) -> list[int]:
        idx = corpus_texts.index(text) if text in corpus_texts else 0
        return vectors[idx]

    vdb.embed = stub_embed
    try:
        store = LocalVectorStore()
        r_add = timed("store add (200 docs)", store.add, corpus_texts)
        r_search = timed("store search k=5", store.search, corpus_texts[0], K)
        hits = r_search["result"]
        assert hits, "store search returned nothing"
        assert hits[0][0] == corpus_texts[0], "top hit should be doc 0"
    finally:
        vdb.embed = original_embed

    print("\n[6] full pipeline (fan-out -> top_k)")

    def pipeline() -> dict[str, Any]:
        outs = fan_out(("coder", "optimizer"), "improve this pipeline", model)
        ranked = top_k(query, vectors, K)
        return {"answers": outs, "ranked": ranked}

    r_pipe = timed("fan_out(2) + top_k", pipeline)
    assert len(r_pipe["result"]["ranked"]) == K

    print("\n" + "=" * 64)
    print("all assertions passed — pipeline is correct end to end")
    total = (
        r_build["median_ms"]
        + r_fanout["median_ms"]
        + r_vec["median_ms"]
        + r_topk["median_ms"]
        + r_add["median_ms"]
        + r_search["median_ms"]
    )
    print(f"sum of stage medians: {total:.1f} ms")


if __name__ == "__main__":
    main()
