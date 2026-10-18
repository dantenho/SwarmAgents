"""Hypothesis property-based tests for kernels and rerank invariants."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from swarmagents.kernels import batch_cosine, cosine_int8, top_k

int8 = st.integers(min_value=-127, max_value=127)
vectors = st.lists(int8, min_size=1, max_size=64)
empty_friendly = st.lists(int8, min_size=0, max_size=32)


@given(a=vectors, b=vectors)
@settings(max_examples=100, deadline=None)
def test_cosine_identity(a, b):
    """cos(v, v) == 1 for any nonzero vector; batch agrees with single."""
    if any(x != 0 for x in a):
        assert abs(cosine_int8(a, a) - 1.0) < 1e-9
    batch = batch_cosine([a], [a])
    assert abs(batch[0] - cosine_int8(a, a)) < 1e-9


@given(a=vectors, b=vectors)
def test_cosine_symmetry(a, b):
    """cos(a, b) == cos(b, a) when lengths match."""
    from hypothesis import assume

    assume(len(a) == len(b))
    assert abs(cosine_int8(a, b) - cosine_int8(b, a)) < 1e-9


@given(a=vectors, b=vectors)
@settings(max_examples=50)
def test_cosine_range(a, b):
    """Similarity always lies in [-1, 1] for matching lengths."""
    from hypothesis import assume

    assume(len(a) == len(b))
    score = cosine_int8(a, b)
    assert -1.0 <= score <= 1.0


@given(a=vectors, b=vectors, k=st.integers(min_value=1, max_value=10))
@settings(max_examples=30, deadline=None)
def test_top_k_invariants(a, b, k):
    """top_k returns sorted, unique, at most k results for equal lengths."""
    from hypothesis import assume

    assume(len(a) == len(b))
    corpus = [a, b]
    results = top_k(a, corpus, k=k)
    assert len(results) <= k
    assert len(results) <= 2
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    indices = [i for i, _ in results]
    assert len(indices) == len(set(indices))
    for i, s in results:
        assert abs(s - cosine_int8(a, corpus[i])) < 1e-9


@given(size=st.integers(min_value=1, max_value=64))
def test_zero_vector_is_zero_similarity(size):
    """A zero vector scores exactly 0.0 against any nonzero vector."""
    zero = [0] * size
    other = [1] * size
    assert cosine_int8(zero, other) == 0.0
    assert cosine_int8(other, zero) == 0.0
