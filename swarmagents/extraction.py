"""Structured extraction with langextract and Neo4jVector-backed retrieval.

Extends the memory and deepresearch specialists with:
- langextract-based structured fact extraction from raw text
- a langchain-neo4j Neo4jVector retriever wired to the int8 embeddings
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

EMBEDDING_DIM = 384


def _have_langextract() -> bool:
    return importlib.util.find_spec("langextract") is not None


def extract_facts(text: str) -> list[dict[str, Any]]:
    """Extract structured facts from text via langextract.

    Args:
        text: Raw text to mine for structured content.

    Returns:
        A list of extracted record dicts; empty list when langextract
        is unavailable or extraction yields nothing.
    """
    if not _have_langextract():
        return []
    from langextract import extract

    try:
        result = extract(text)
    except Exception:
        return []
    if isinstance(result, dict) and "extraction" in result:
        records = result["extraction"]
        if isinstance(records, dict):
            return [dict(records)]
        return [dict(r) for r in records if isinstance(r, dict)]
    if isinstance(result, list):
        return [dict(r) for r in result if isinstance(r, dict)]
    if isinstance(result, dict):
        return [dict(result)]
    return []


def neo4j_retriever(
    index_name: str = "swarm_chunks",
    node_label: str = "Chunk",
) -> Any | None:
    """Build a Neo4jVector retriever over the int8 embedding pipeline.

    Requires NEO4J_URI/USERNAME/PASSWORD plus SUPABASE-independent local
    embeddings. Returns None when Neo4j credentials are absent.

    Args:
        index_name: Neo4j vector index name.
        node_label: Label of nodes holding the chunks.

    Returns:
        A configured Neo4jVector retriever, or None without credentials.
    """
    url = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USERNAME", "")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not (url and user and password):
        return None
    from langchain_core.embeddings import Embeddings as CoreEmbeddings

    class Int8Embeddings(CoreEmbeddings):
        """MiniLM int8 embeddings adapted to the LangChain interface."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            from swarmagents.embeddings import embed_batch

            vectors = embed_batch(texts)
            return [[float(x) for x in v] for v in vectors]

        def embed_query(self, text: str) -> list[float]:
            from swarmagents.embeddings import embed

            return [float(x) for x in embed(text)]

    from langchain_neo4j import Neo4jVector
    from langchain_neo4j.vectorstores.neo4j_vector import DistanceStrategy

    return Neo4jVector(
        Int8Embeddings(),
        url=url,
        username=user,
        password=password,
        index_name=index_name,
        node_label=node_label,
        embedding_dimension=EMBEDDING_DIM,
        distance_strategy=DistanceStrategy.COSINE,
    )


__all__ = ("extract_facts", "neo4j_retriever")
