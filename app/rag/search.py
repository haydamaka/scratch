"""
Entry point for table search: a dense (semantic) branch and a lexical one,
fused into a single ranked list.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from typing import Optional

from app.core.logger import get_logger
from app.rag.embedding_vertex import get_vectordb_embedding_fn
from app.rag.vector_store import get_collection, persona_filter

logger = get_logger(__name__)

#Used only for table catalog, but keeping this configurable in case we go for other collection
COLLECTION_NAME = "table_catalog"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchConfig:
    """Every search knob, with its default. Override per call via ``search()``."""

    # —— how many records ————————————————————————————————————————
    # Recall measured on the sweep: @5=0.310 @10=0.448 @20=0.586 @30=0.586.
    # Five returns a third of the ground-truth tables; the curve is flat past 20.
    top_n: int = 30                     # when the caller does not say
    # How deep each branch is read before fusion. Deeper than any caller asks for,
    # so fusion has material to choose between rather than merging two lists that
    # were already cut. One number, because nothing has ever tuned two.
    candidate_n: int = 150

    # —— retriever weights (query time) ————————————————————————————————
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0

    # —— index build (changing these rebuilds the keyword index) ——————
    # bm25_k1 controls term-frequency saturation; bm25_b controls how much
    # document length normalization affects the score.
    bm25_k1: float = 1.2
    # 0.0 = no document-length normalisation. Measured over 125 questions:
    # b=0.0 and b=0.4 both return 122 of 161 tables, the standard 0.75 returns
    # 114. One document here holds a three-word table name and a two-hundred-name
    # column list, so normalising by length penalises exactly the rows carrying
    # the most evidence. Off is both the best measured setting and one fewer
    # thing to tune.
    bm25_b: float = 0.0
    keyword_fields: frozenset[str] = frozenset(
        {"name", "alias", "domain", "description", "rules", "columns"}
    )


    @classmethod
    def from_env(cls) -> "SearchConfig":
        """Overlay environment variables onto the defaults above."""
        values = {}
        for field in fields(cls):
            raw = os.getenv(_ENV_NAMES.get(field.name, field.name.upper()))
            if raw is not None:
                values[field.name] = _PARSE[field.type](raw)
        return cls(**values)

    def with_overrides(self, **overrides) -> "SearchConfig":
        """Copy with the non-``None`` overrides applied; unknown names raise."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


# Field name → environment variable, where the two differ. Every name here is
# one already in use; do not rename them, deployments set them.
_ENV_NAMES = {
    "top_n":             "SEARCH_TOP_N",
    "candidate_n":       "KEYWORD_CANDIDATE_N",
    "rrf_k":             "KEYWORD_RRF_K",
    "keyword_fields":    "KEYWORD_FIELDS",
}

_PARSE = {
    "int":   int,
    "float": float,
    "tuple[int, int]":  lambda s: tuple(int(x) for x in s.split(",")),
    "frozenset[str]":   lambda s: frozenset(f.strip() for f in s.split(",") if f.strip()),
}

_CONFIG: Optional[SearchConfig] = None


def get_search_config() -> SearchConfig:
    """The process-wide default config, resolved from the environment once."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = SearchConfig.from_env()
    return _CONFIG


def set_search_config(cfg: SearchConfig) -> None:
    """Replace the process-wide config
    """
    global _CONFIG
    _CONFIG = cfg


# ---------------------------------------------------------------------------
# Result fusion helpers
# ---------------------------------------------------------------------------

def fuse_rrf(
    a_ids: list[str],
    b_ids: list[str],
    k: int = 60,
    w_a: float = 1.0,
    w_b: float = 1.0,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion over two ranked lists.

    Each document accumulates ``weight / (k + rank)`` per retriever (1-based rank).
    Returns ``(id, fused_score)`` pairs sorted descending.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(a_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_a / (k + rank)
    for rank, doc_id in enumerate(b_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_b / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def dense_retrieve(collection, query: str, n: int, where: Optional[dict]) -> dict:
    """Dense (semantic) retriever: ``{id: (meta, document, distance)}``, best first."""
    embedding = get_vectordb_embedding_fn(task="RETRIEVAL_QUERY")([query])[0]
    res = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
        **({"where": where} if where else {}),
    )
    return {
        meta["table_name"]: (meta, doc, dist)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    }


def _sparse_retrieve(
    collection,
    query: str,
    n: int,
    persona_id: Optional[int],
    cfg: SearchConfig,
) -> "tuple[list[str], dict[str, float]]":
    """Sparse (lexical) retriever: ``(ids, scores)``.

    Raises when the index is unusable. Falling back to the dense branch alone
    would answer at measurably worse recall — 0.379 against 0.586 on the sweep —
    while still looking like a success to every caller, and the fault is never
    transient: it repeats on every query until the index is rebuilt. A request
    that fails is the only version of this anyone notices.
    """
    #imported here to avoid cyclic imports
    from app.rag.keyword_index import get_keyword_index_service

    try:
        svc = get_keyword_index_service()
        # Startup builds this (vectordb_loader). Here only for flows that never
        # start the loader: CLIs, tests, the eval harness.
        svc.warm(collection=collection, cfg=cfg)
        idx = svc.get()
    except Exception as exc:
        raise RuntimeError(
            "Keyword index could not be built; refusing to answer from the dense "
            "branch alone."
        ) from exc

    if idx is None:
        raise RuntimeError(
            "Keyword index is not built; refusing to answer from the dense branch alone."
        )

    hits = idx.search(query, top_n=n, persona_id=persona_id)
    return [h[0] for h in hits], dict(hits)


def _combine_rankings(
    dense_ids: list[str],
    sparse_ids: list[str],
    cfg: SearchConfig,
) -> "tuple[list[str], dict[str, float]]":
    """``(ranked_ids, score_by_id)`` — both branches, always fused.

    A weight of zero needs no special case: it contributes ``0 / (k + rank)`` to
    every document, which leaves the other branch's ranking intact and sinks the
    documents only this branch found to the bottom of the list.
    """
    fused = fuse_rrf(dense_ids, sparse_ids, k=cfg.rrf_k,
                     w_a=cfg.vector_weight, w_b=cfg.keyword_weight)
    return [fid for fid, _ in fused], dict(fused)


def _fetch_missing_records(
    collection, all_ids: list[str], fetched_records: dict, where: Optional[dict],
) -> None:
    """Fetch metadata for ids the dense (semantic) retriever didn't saw among the top-n.
    The ``where`` filter used to match by persona
    """
    missing = [tid for tid in all_ids if tid not in fetched_records]
    if not missing:
        return
    try:
        fetched = collection.get(
            ids=missing,
            include=["documents", "metadatas"],
            **({"where": where} if where else {}),
        )
        for fid, meta, doc in zip(
            fetched.get("ids") or [],
            fetched.get("metadatas") or [],
            fetched.get("documents") or [],
        ):
            fetched_records[fid] = (meta, doc, _NO_DISTANCE)
    except Exception as exc:
        logger.warning("[table_info] Could not fetch kw-only metadata: %s", exc)


# Sentinel distance for a hit the dense (semantic) retriever never returned
_NO_DISTANCE = 9999.0


def search(
    query: str,
    top_n: Optional[int] = None,
    persona_id: Optional[int] = None,
    *,
    config: Optional[SearchConfig] = None,
    **overrides,
) -> list[dict]:
    """Run hybrid table search and return ranked hit dicts.

    Args:
        query:      Natural-language user question.
        top_n:      Maximum number of tables to return; ``cfg.top_n`` when unset.
        persona_id: Restrict to tables tagged with this persona.
        config:     Base config; the process default when unset.
        overrides:  Any :class:`SearchConfig` field, for this call only.

    Returns:
        Hit dicts with ``table_name``, ``table_description``, ``domain_mapping``,
        ``persona_id``, ``table_alias``, ``distance``, ``document``, plus the
        provenance fields ``vector_rank``, ``keyword_rank``, ``keyword_score``,
        ``fused_score`` and ``match_source``.
    """
    cfg = (config or get_search_config()).with_overrides(**overrides)
    if not top_n or top_n <= 0:
        top_n = cfg.top_n

    from app.rag.vectordb_loader import wait_until_ready
    # Return deliberately ignored: on timeout (or a failed load) we serve the
    # persisted store rather than refuse the request, so the answer may come from
    # the previous catalog. The cost is on the sparse arm — `_sparse_retrieve`
    # warms the index itself, so a query arriving mid-load fits BM25 to a
    # half-written catalog and caches that until restart.
    wait_until_ready()

    collection = get_collection(
        COLLECTION_NAME, get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT"),
    )
    n_total = collection.count()
    if n_total == 0:
        # Nothing loaded is a fault; nothing found is an empty list. An empty
        # collection cannot be told apart from "no table matches" by the caller,
        # so it is raised rather than returned.
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' is empty — the loader has not run."
        )

    where = persona_filter(persona_id)
    logger.info(
        '[table_info] Hybrid search: "%s…" (top_n=%d, persona_id=%s)',
        query[:60], top_n, persona_id,
    )

    # Never shallower than what the caller will return.
    candidate_n = max(cfg.candidate_n, top_n)
    records = dense_retrieve(collection, query, min(candidate_n, n_total), where)
    dense_ids = list(records)

    sparse_ids, sparse_scores = _sparse_retrieve(
        collection, query, candidate_n, persona_id, cfg,
    )

    ranked_ids, fused_scores = _combine_rankings(dense_ids, sparse_ids, cfg)
    final_ids = ranked_ids[:top_n]

    _fetch_missing_records(collection, final_ids, records, where)

    dense_rank = {tid: i for i, tid in enumerate(dense_ids, start=1)}
    sparse_rank = {tid: i for i, tid in enumerate(sparse_ids, start=1)}

    hits: list[dict] = []
    for tid in final_ids:
        entry = records.get(tid)
        if entry is None:
            continue
        meta, document, distance = entry
        in_dense, in_sparse = tid in dense_rank, tid in sparse_rank
        hits.append({
            "table_name":        meta["table_name"],
            "table_description": meta["table_description"],
            "domain_mapping":    meta.get("domain_mapping", ""),
            "persona_id":        meta.get("persona_id", -1),
            "table_alias":       json.loads(meta.get("table_alias", "[]")),
            "distance":          round(distance, 4),
            "document":          document,
            "vector_rank":       dense_rank.get(tid),
            "keyword_rank":      sparse_rank.get(tid),
            "keyword_score":     round(sparse_scores.get(tid, 0.0), 6),
            "fused_score":       round(fused_scores.get(tid, 0.0), 6),
            "match_source":      "both" if in_dense and in_sparse else ("keyword" if in_sparse else "vector"),
        })
    return hits
