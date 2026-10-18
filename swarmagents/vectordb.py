"""Vector store backends: Supabase pgvector remote, local in-memory fallback.

All heavy imports are lazy (PEP 810 style). The Supabase client requires
``SUPABASE_URL`` and ``SUPABASE_KEY`` in the environment; without them the
store falls back to a local in-memory index scored by the kernels dispatcher.
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING, Any, Protocol

from swarmagents.embeddings import EMBEDDING_DIM, embed
from swarmagents.kernels import top_k

if TYPE_CHECKING:
    Vector = list[int]


class VectorStore(Protocol):
    """Interface for vector stores."""

    def add(
        self, texts: list[str], metadata: list[dict[str, Any]] | None = None
    ) -> list[str]:
        """Embed and store texts, returning their ids."""
        ...

    def search(self, query: str, k: int = 5) -> list[tuple[str, float, dict[str, Any]]]:
        """Return the k nearest (text, score, metadata) triples for a query."""
        ...


def _have_supabase() -> bool:
    return importlib.util.find_spec("supabase") is not None


class SupabaseVectorStore:
    """pgvector-backed store on Supabase (requires env credentials).

    Table schema expected (create once via SQL editor):

        create extension if not exists vector;
        create table if not exists swarm_docs (
            id bigserial primary key,
            content text not null,
            metadata jsonb not null default '{}',
            embedding vector(384) not null
        );
    """

    def __init__(self, table: str = "swarm_docs") -> None:
        """Initialize the store against the configured Supabase project.

        Args:
            table: Postgres table name holding documents and embeddings.
        """
        if not _have_supabase():
            raise ImportError(
                "install the 'supabase' package to use SupabaseVectorStore"
            )
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        from supabase import create_client

        self._client = create_client(url, key)
        self._table = table

    def add(self, texts, metadata=None) -> list[str]:
        """Embed texts and insert them with their pgvector embeddings.

        Args:
            texts: Documents to embed and store.
            metadata: Optional per-document metadata, aligned with texts.

        Returns:
            The inserted row ids as strings.
        """
        vectors = [embed(t) for t in texts]
        rows = [
            {
                "content": t,
                "metadata": (metadata[i] if metadata else {}),
                "embedding": v,
            }
            for i, (t, v) in enumerate(zip(texts, vectors, strict=True))
        ]
        result = self._client.table(self._table).insert(rows).execute()
        return [str(r["id"]) for r in result.data if isinstance(r, dict)]

    def search(self, query: str, k: int = 5):
        """Cosine-search the table for the k nearest documents.

        Args:
            query: Query text to embed and match.
            k: Number of neighbors to return.

        Returns:
            The k nearest (text, score, metadata) triples, descending.
        """
        vector = embed(query)
        rpc = self._client.rpc(
            "match_swarm_docs",
            {
                "query_embedding": vector,
                "match_count": k,
                "p_table": self._table,
            },
        )
        data = rpc.execute().data
        items: list[Any] = list(data) if isinstance(data, list) else []
        rows: list[dict[str, Any]] = [dict(r) for r in items if isinstance(r, dict)]
        return [
            (
                str(r["content"]),
                float(r["similarity"]),
                dict(r.get("metadata") or {}),
            )
            for r in rows
        ]


class LocalVectorStore:
    """In-memory store scored by the adaptive kernels dispatcher."""

    def __init__(self) -> None:
        """Initialize an empty local store."""
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._vectors: list[Vector] = []

    def add(self, texts, metadata=None) -> list[str]:
        """Embed and retain texts in memory.

        Args:
            texts: Documents to embed and store.
            metadata: Optional per-document metadata, aligned with texts.

        Returns:
            Positional ids as strings.
        """
        start = len(self._texts)
        for i, t in enumerate(texts):
            self._texts.append(t)
            self._metadata.append(metadata[i] if metadata else {})
            self._vectors.append(embed(t))
        return [str(start + i) for i in range(len(texts))]

    def search(self, query: str, k: int = 5):
        """Cosine-search in memory via the kernels dispatcher.

        Args:
            query: Query text to embed and match.
            k: Number of neighbors to return.

        Returns:
            The k nearest (text, score, metadata) triples, descending.
        """
        if not self._texts:
            return []
        vector = embed(query, use_cache=True)
        ranked = top_k(vector, self._vectors, k=k)
        return [(self._texts[i], score, self._metadata[i]) for i, score in ranked]


def create_vector_store() -> VectorStore:
    """Return a Supabase store when configured, else the local fallback.

    Returns:
        A VectorStore instance (Supabase when SUPABASE_URL/KEY are set).
    """
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY") and _have_supabase():
        try:
            return SupabaseVectorStore()
        except Exception:
            pass
    return LocalVectorStore()


__all__ = (
    "EMBEDDING_DIM",
    "LocalVectorStore",
    "SupabaseVectorStore",
    "VectorStore",
    "create_vector_store",
)
