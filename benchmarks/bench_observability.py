"""Instrumented benchmark: OpenTelemetry + Prometheus + loguru around kernels.

Measures the overhead of the observability stack while running the int8
similarity workload, and exports Prometheus metrics on demand.

Run:
    PYTHONPATH=. .venv/bin/python benchmarks/bench_observability.py
    PROMETHEUS_PORT=8000 ...   # to expose /metrics
"""

from __future__ import annotations

import os
import random
import statistics
import time
from typing import Any

DIM = 384
CORPUS = 2_000
K = 10
REPEAT = 20


def _setup_loguru() -> Any:
    from loguru import logger

    logger.remove()
    logger.add(lambda msg: None, level="INFO")
    return logger


def _setup_prometheus() -> dict[str, Any]:
    from prometheus_client import Counter, Histogram, Summary

    return {
        "queries": Counter("swarm_sim_queries_total", "Similarity queries executed"),
        "latency": Histogram(
            "swarm_sim_latency_seconds",
            "Similarity call latency",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
        ),
        "score": Summary("swarm_sim_top_score", "Top similarity score observed"),
    }


def _setup_otel() -> Any:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    trace.set_tracer_provider(TracerProvider())
    return trace.get_tracer("swarmagents.bench")


def _run_corpus(rng: random.Random) -> tuple[list[list[int]], list[int]]:
    corpus = [[rng.randint(-127, 127) for _ in range(DIM)] for _ in range(CORPUS)]
    query = [rng.randint(-127, 127) for _ in range(DIM)]
    return corpus, query


def main() -> None:
    """Run the instrumented benchmark and report per-stack overhead."""
    from swarmagents.kernels import active_backend, top_k

    logger = _setup_loguru()
    prom = _setup_prometheus()
    tracer = _setup_otel()

    rng = random.Random(42)
    corpus, query = _run_corpus(rng)

    print(f"kernels backend: {active_backend()}")
    print(f"workload: top-{K} over {CORPUS}x{DIM} int8, repeat {REPEAT}")
    print("-" * 56)

    def bare() -> list[tuple[int, float]]:
        return top_k(query, corpus, k=K)

    def with_loguru() -> list[tuple[int, float]]:
        logger.info("rerank start k={}", K)
        out = top_k(query, corpus, k=K)
        logger.info("rerank done top1={}", out[0][1])
        return out

    def with_prometheus() -> list[tuple[int, float]]:
        with prom["latency"].time():
            prom["queries"].inc()
            out = top_k(query, corpus, k=K)
        prom["score"].observe(out[0][1])
        return out

    def with_otel() -> list[tuple[int, float]]:
        with tracer.start_as_current_span("rerank") as span:
            span.set_attribute("corpus.size", CORPUS)
            span.set_attribute("rerank.k", K)
            out = top_k(query, corpus, k=K)
            span.set_attribute("rerank.top1_score", out[0][1])
            return out

    def with_all() -> list[tuple[int, float]]:
        logger.info("instrumented rerank start")
        with tracer.start_as_current_span("rerank"):
            with prom["latency"].time():
                prom["queries"].inc()
                out = top_k(query, corpus, k=K)
        prom["score"].observe(out[0][1])
        logger.info("instrumented rerank done")
        return out

    def avg_ms(fn: Any, repeat: int = REPEAT) -> float:
        samples: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000)
        return statistics.mean(samples)

    expected = bare()
    for wrapped in (with_loguru, with_prometheus, with_otel, with_all):
        got = wrapped()
        assert got[0][0] == expected[0][0], "instrumentation changed results"

    t_bare = avg_ms(bare)
    print(f"{'bare':24s} avg {t_bare:9.3f} ms")
    for label, fn in (
        ("+ loguru", with_loguru),
        ("+ prometheus", with_prometheus),
        ("+ opentelemetry", with_otel),
        ("+ all three", with_all),
    ):
        t = avg_ms(fn)
    print(f"{'':24s} (per-stack averages below)")
    rows = {
        "bare": t_bare,
        "+ loguru": avg_ms(with_loguru),
        "+ prometheus": avg_ms(with_prometheus),
        "+ opentelemetry": avg_ms(with_otel),
        "+ all three": avg_ms(with_all),
    }
    print("-" * 56)
    for label, t in rows.items():
        overhead = (t - t_bare) * 1000
        print(f"{label:24s} avg {t:9.3f} ms  overhead {overhead:+8.1f} us/call")

    port = os.getenv("PROMETHEUS_PORT")
    if port:
        from prometheus_client import start_http_server

        start_http_server(int(port))
        print(f"\nprometheus /metrics exposed on :{port}")
    else:
        from prometheus_client import generate_latest

        metrics = generate_latest().decode()
        shown = "\n".join(
            line for line in metrics.splitlines() if line.startswith("swarm_sim")
        )
        print("\nprometheus metrics (text format):")
        print(shown or "(no samples recorded)")


if __name__ == "__main__":
    main()
