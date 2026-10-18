"""Free-threaded Python vs Numba njit benchmark for int8 kernels.

Run on a free-threaded (no-GIL) CPython build:
    PYTHON_GIL=0 .venv-ft/bin/python benchmarks/bench_freethread.py
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import timeit
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

DIM = 384
N_BATCH = 10_000
N_THREADS = 8


def _lazy_njit():
    if importlib.util.find_spec("numba") is None:
        return None
    from numba import njit

    return njit


def cosine_int8_python(a: list[int], b: list[int]) -> float:
    """Pure-Python int8 cosine similarity."""
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


def run_python_threaded(batch_a: list[list[int]], batch_b: list[list[int]]) -> None:
    """Run the batch split across OS threads (requires free-threading)."""
    chunk = len(batch_a) // N_THREADS
    [threading.Barrier(N_THREADS) for _ in range(N_THREADS)]

    def worker(idx: int) -> None:
        lo = idx * chunk
        hi = lo + chunk if idx < N_THREADS - 1 else len(batch_a)
        for i in range(lo, hi):
            cosine_int8_python(batch_a[i], batch_b[i])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_python_sequential(batch_a, batch_b) -> None:
    for x, y in zip(batch_a, batch_b, strict=True):
        cosine_int8_python(x, y)


def _bench(label: str, fn, *args) -> float:
    best = min(timeit.repeat(lambda: fn(*args), number=1, repeat=3))
    unit = "us" if best < 0.001 else "ms"
    value = best * 1000 if unit == "ms" else best * 1_000_000
    print(f"{label:28s} {value:12.3f} {unit}")
    return best


def main() -> None:
    print(f"python: {sys.version.split()[0]}  GIL enabled: {sys._is_gil_enabled()}")
    import numpy as np

    rng = np.random.default_rng(42)
    a_np = rng.integers(-127, 128, size=DIM, dtype=np.int8)
    b_np = rng.integers(-127, 128, size=DIM, dtype=np.int8)
    a_list = [int(x) for x in a_np]
    b_list = [int(x) for x in b_np]
    batch_a = [a_list] * N_BATCH
    batch_b = [b_list] * N_BATCH

    njit = _lazy_njit()

    print(f"\nint8 cosine, {DIM}-dim, batch {N_BATCH}")
    print("-" * 48)

    t_seq = _bench("python sequential", run_python_sequential, batch_a, batch_b)
    t_thr = _bench(f"python {N_THREADS} threads", run_python_threaded, batch_a, batch_b)

    if njit is not None:
        from numba import njit as njit_dec
        from numba import prange

        @njit_dec(cache=True, parallel=True)
        def batch_numba(a, b):
            out = np.empty(a.shape[0])
            for i in prange(a.shape[0]):
                dot = 0
                na = 0
                nb = 0
                for j in range(a.shape[1]):
                    dot += a[i, j] * b[i, j]
                    na += a[i, j] * a[i, j]
                    nb += b[i, j] * b[i, j]
                out[i] = dot / ((na**0.5) * (nb**0.5))
            return out

        big_a = np.tile(a_np, (N_BATCH, 1))
        big_b = np.tile(b_np, (N_BATCH, 1))
        batch_numba(big_a, big_b)
        t_numba = _bench("numba njit parallel", batch_numba, big_a, big_b)
    else:
        t_numba = None

    print("\nspeedups vs python sequential:")
    print(f"  free-threaded ({N_THREADS} threads): {t_seq / t_thr:8.1f}x")
    if t_numba is not None:
        print(f"  numba njit parallel:            {t_seq / t_numba:8.1f}x")


if __name__ == "__main__":
    main()
