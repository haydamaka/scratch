"""Polymorphic vector-store abstraction.
providing factory method for getting vector store (currently chromadb or milvus)
and the abstract methods
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence

# ``EmbeddingFunction`` is a chromadb type but is only ever used here as a
# callable ``fn(list[str]) -> list[list[float]]``; typing it loosely keeps this
# module import-light and provider-neutral.
EmbeddingFn = Any

# Vector db provider selector environment variable.
VECTORDB_PROVIDER_ENV = "VECTORDB_PROVIDER"


class VectorDbProvider(str, Enum):
    """Supported vector db providers."""

    CHROMADB = "chromadb"
    MILVUS = "milvus"


_DEFAULT_PROVIDER = VectorDbProvider.CHROMADB


class VectorCollection(ABC):
    """Provider-agnostic handle to a single named collection."""

    name: str

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored documents."""

    @abstractmethod
    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        """Insert or replace documents (embedding them via the backend)."""

    @abstractmethod
    def get(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        include: Optional[Sequence[str]] = None,
    ) -> dict:
        """Fetch stored rows by id. Returns a dict with at least ``ids`` and
        (when requested) ``metadatas`` / ``documents`` keys."""

    @abstractmethod
    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        include: Optional[Sequence[str]] = None,
        where: Optional[dict] = None,
    ) -> dict:
        """Nearest-neighbour search. Returns ChromaDB-shaped nested lists:
        ``{"ids": [[...]], "documents": [[...]], "metadatas": [[...]],
        "distances": [[...]]}``."""


class VectorStore(ABC):
    """Backend-agnostic vector-store client/factory."""

    @abstractmethod
    def get_or_create_collection(
        self,
        name: str,
        embedding_fn: EmbeddingFn,
    ) -> VectorCollection:
        """Return the named collection, creating it if absent."""

    @abstractmethod
    def get_collection(
        self,
        name: str,
        embedding_fn: EmbeddingFn,
    ) -> VectorCollection:
        """Return the existing named collection (raise/behave per backend if
        missing)."""


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

_store_instances: "dict[VectorDbProvider, VectorStore]" = {}


def _selected_provider() -> VectorDbProvider:
    value = os.getenv(VECTORDB_PROVIDER_ENV)
    if not value:
        return _DEFAULT_PROVIDER
    try:
        return VectorDbProvider(value.strip().lower())
    except ValueError:
        raise ValueError(
            f"Unknown {VECTORDB_PROVIDER_ENV}={value!r}. "
            f"Expected one of: "
            f"{', '.join(p.value for p in VectorDbProvider)}."
        )


def get_vector_store() -> VectorStore:
    """Return a cached :class:`VectorStore` for the vector db provider named by
    ``$VECTORDB_PROVIDER`` (defaults to ChromaDB)."""
    provider = _selected_provider()

    if provider not in _store_instances:
        if provider is VectorDbProvider.CHROMADB:
            from app.rag.chroma_db import ChromaVectorStore
            _store_instances[provider] = ChromaVectorStore()
        elif provider is VectorDbProvider.MILVUS:
            from app.rag.milvus_db import MilvusVectorStore
            _store_instances[provider] = MilvusVectorStore()
    return _store_instances[provider]



def get_or_create_collection(
    name: str,
    embedding_fn: EmbeddingFn,
) -> VectorCollection:
    """Provider-agnostic ``get_or_create_collection`` via the selected store."""
    return get_vector_store().get_or_create_collection(name, embedding_fn)


def get_collection(
    name: str,
    embedding_fn: EmbeddingFn,
) -> VectorCollection:
    """Provider-agnostic ``get_collection`` via the selected store."""
    return get_vector_store().get_collection(name, embedding_fn)


def store_location() -> str:
    """Human-readable description of the active vector store's storage location.

    Provider-neutral: reports the selected provider plus its on-disk / server
    location (used purely for logging by the loaders)."""
    provider = _selected_provider()
    storage_path = os.getenv("VECTORDB_STORAGE_PATH") or ""

    if provider is VectorDbProvider.MILVUS:
        location = os.getenv("MILVUS_URI") or os.getenv("MILVUS_LITE_PATH") or storage_path
    else:
        location = storage_path

    return f"{provider.value} @ {location}" if location else provider.value




# ─────────────────────────────────────────────
# Metadata filters
# ─────────────────────────────────────────────

def persona_filter(persona_id: Optional[int]) -> Optional[dict]:
    """``where`` filter matching every row tagged with ``persona_id``.

    The loaders' ``_build_metadata`` write a ``persona_id_<pid>`` flag per
    persona, plus a scalar ``persona_id`` holding only the *first* one. The flags
    are what the tag list means; filtering on the scalar hides a row tagged
    ``[2, 5]`` from a ``persona_id=5`` search.

    Written in the ChromaDB filter dialect, which is this module's contract —
    ``milvus_db._translate_where`` converts it for the other backend.
    """
    if persona_id is None:
        return None
    return {f"persona_id_{int(persona_id)}": {"$eq": 1}}
