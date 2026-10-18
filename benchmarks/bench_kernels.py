"""Benchmarks: pure Python (no JIT) vs Numba njit vs pyo3/Rust for int8 kernels."""

from __future__ import annotations

import timeit
from typing import Any, Callable

import numpy as np

DIM = 384
N_BATCH = 10_000


def cosine_int8_python(a: list[int], b: list[int]) -> float:
    """Compute int8 cosine similarity in pure Python (no JIT).

    Args:
        a: First vector.
        b: Second vector of the same length.

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
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


def _run_benchmark(label: str, fn: Callable[..., Any], *args: Any) -> float:
    """Time fn and print the best-of-5 duration.

    Args:
        label: Display name for the benchmark row.
        fn: Callable to measure.
        *args: Arguments forwarded to fn.

    Returns:
        Best (minimum) duration in seconds.
    """
    best = min(timeit.repeat(lambda: fn(*args), number=1, repeat=5))
    unit = "us" if best < 0.001 else "ms"
    value = best * 1000 if unit == "ms" else best * 1_000_000
    print(f"{label:24s} {value:12.3f} {unit}")
    return best


def main() -> None:
    """Run all kernel benchmarks and print per-backend speedups."""
    rng = np.random.default_rng(42)
    a_np = rng.integers(-127, 128, size=DIM, dtype=np.int8)
    b_np = rng.integers(-127, 128, size=DIM, dtype=np.int8)
    a_list = [int(x) for x in a_np]
    b_list = [int(x) for x in b_np]

    try:
        from numba import njit, prange

        @njit(cache=True)
        def cosine_int8_numba(a, b):
            dot = 0
            na = 0
            nb = 0
            for i in range(a.shape[0]):
                dot += a[i] * b[i]
                na += a[i] * a[i]
                nb += b[i] * b[i]
            if na == 0 or nb == 0:
                return 0.0
            return dot / ((na**0.5) * (nb**0.5))

        cosine_int8_numba(a_np, b_np)
    except ImportError:
        cosine_int8_numba = None

    cosine_rust: Any
    try:
        from swarmagents_core import cosine_similarity_int8 as cosine_rust  # type: ignore[attr-defined]
    except ImportError:
        cosine_rust = None

    print(f"int8 cosine similarity, {DIM}-dim vectors")
    print("-" * 40)
    t_python = _run_benchmark("python (no jit)", cosine_int8_python, a_list, b_list)

    if cosine_int8_numba is not None:
        t_numba = _run_benchmark("numba @njit", cosine_int8_numba, a_np, b_np)
    else:
        t_numba = None
    if cosine_rust is not None:
        t_rust = _run_benchmark("pyo3/rust", cosine_rust, a_list, b_list)
    else:
        t_rust = None

    print()
    print(f"batch: {N_BATCH} pairs, {DIM}-dim")
    print("-" * 40)
    batch_a = [a_list] * N_BATCH
    batch_b = [b_list] * N_BATCH

    def python_batch() -> None:
        """Score the whole batch sequentially in pure Python."""
        for x, y in zip(batch_a, batch_b, strict=True):
            cosine_int8_python(x, y)

    t_py_batch = _run_benchmark("python (no jit)", python_batch)

    if cosine_int8_numba is not None:
        big_a = np.tile(a_np, (N_BATCH, 1))
        big_b = np.tile(b_np, (N_BATCH, 1))

        @njit(cache=True, parallel=True)
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

        batch_numba(big_a, big_b)
        t_numba_batch = _run_benchmark(
            "numba @njit parallel", batch_numba, big_a, big_b
        )
    else:
        t_numba_batch = None

    if cosine_rust is not None:

        def rust_batch() -> None:
            """Score the whole batch via the pyo3 Rust extension."""
            for x, y in zip(batch_a, batch_b, strict=True):
                cosine_rust(x, y)

        t_rust_batch = _run_benchmark("pyo3/rust", rust_batch)
    else:
        t_rust_batch = None

    print()
    print("speedups vs pure python:")
    if t_numba is not None:
        print(f"  numba single:  {t_python / t_numba:8.1f}x")
    if t_rust is not None:
        print(f"  rust single:   {t_python / t_rust:8.1f}x")
    if t_numba_batch is not None:
        print(f"  numba batch:   {t_py_batch / t_numba_batch:8.1f}x")
    if t_rust_batch is not None:
        print(f"  rust batch:    {t_py_batch / t_rust_batch:8.1f}x")


if __name__ == "__main__":
    main()
