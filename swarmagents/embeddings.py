"""MiniLM-L6-v2 int8 embeddings (384-dim) with Redis caching.

Heavy imports are deferred until first use (PEP 810 lazy imports) so that
importing this module stays cheap.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis import Redis


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_TTL = int(os.getenv("SWARM_CACHE_TTL", "3600"))

_model: Any = None
_redis_client: "Redis | None" = None


def _cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"swarmagents:emb:minilm6:{digest}"


def _get_redis() -> "Redis | None":
    global _redis_client
    if _redis_client is None:
        url = os.getenv("SWARM_REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis  # lazy import

            client = redis.Redis.from_url(url)
            client.ping()
            _redis_client = client
        except Exception:
            return None
    return _redis_client


def _get_model() -> Any:
    global _model
    if _model is None:
        import torch  # lazy import
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        model.eval()
        model.half()
        model = model.to("cpu")
        quant = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        _model = quant
    return _model


def _encode_int8(text: str) -> list[int]:
    import numpy as np  # lazy import
    import torch  # lazy import

    model = _get_model()
    with torch.no_grad():
        emb = model.encode(text, convert_to_numpy=True)
    vec = np.asarray(emb, dtype=np.float32)
    if vec.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"expected {EMBEDDING_DIM}-dim vector, got {vec.shape[0]}")
    scaled = np.clip(np.round(vec * 127.0 / 3.0), -127, 127)
    return [int(x) for x in scaled.astype(np.int8).tolist()]


def embed(text: str, use_cache: bool = True) -> list[int]:
    """Encode text to a 384-dim int8 vector, cached in Redis when available."""
    key = _cache_key(text)
    if use_cache:
        client = _get_redis()
        if client is not None:
            cached = client.get(key)
            if cached is not None:
                import json  # lazy import

                return json.loads(cached)

    vector = _encode_int8(text)

    if use_cache:
        client = _get_redis()
        if client is not None:
            import json  # lazy import

            client.set(key, json.dumps(vector), ex=DEFAULT_TTL)
    return vector


def embed_batch(texts: list[str], use_cache: bool = True) -> list[list[int]]:
    """Encode multiple texts to int8 vectors."""
    return [embed(t, use_cache=use_cache) for t in texts]
