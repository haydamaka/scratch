"""
Entry point for table search: a dense retriever, several sparse
retrievers and the relationship-graph step, orchestrated into one ranked list.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields, replace
from typing import Optional

from app.core.logger import get_logger
from app.rag.embedding_vertex import get_vectordb_embedding_fn
from app.rag.er_filter import select_with_related_tables
from app.rag.query_prep import expand_lexical_query
from app.rag.vector_store import get_collection, persona_filter

logger = get_logger(__name__)

COLLECTION_NAME = "table_catalog"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchConfig:
    """Every search knob, with its default. Override per call via ``search()``."""

    # —— how many records ————————————————————————————————————————
    top_n: int = 5                      # when the caller does not say
    candidate_mult: int = 5             # per-retriever fetch depth = top_n × this …
    candidate_min: int = 50             # … but never shallower than this

    # —— retriever weights (query time) ————————————————————————————————
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    ngram_weight: float = 0.3           # also build-time: 0.0 skips the retriever
    name_alias_weight: float = 1.0
    column_weight: float = 1.0          # also build-time: 0.0 skips the retriever
    name_ngram_weight: float = 0.3      # also build-time: 0.0 skips the retriever

    # —— index build (changing these rebuilds the keyword index) ——————
    bm25_k1: float = 1.2
    bm25_b: float = 0.4                 # below the 0.75 default: short, uneven docs
    # Annotations here are unquoted on purpose: `from_env` dispatches on the
    # annotation text, and `from __future__ import annotations` already keeps
    # them as strings — quoting one would store it with its quotes.
    ngram_range: tuple[int, int] = (3, 5)
    keyword_fields: frozenset[str] = frozenset(
        {"name", "alias", "domain", "description", "rules"}
    )

    # —— relationship-graph expansion (see er_filter.py) ————————————
    # The combined ranking is read far deeper than it is returned:
    # `er_candidates_n` tables are considered, the top `er_anchor_n` are kept
    # unconditionally, and the rest are added only where the graph joins them to
    # one of those. `er_candidates_n = 0` skips the step entirely.
    er_candidates_n: int = 80
    er_anchor_n: int = 10
    er_related_n: int = 20
    # Scoring for the added tables. The three boosts sum to `er_weight_rank` on
    # purpose: everything going right for a candidate can at most fully
    # compensate for being ranked last, and never more.
    er_weight_rank: float = 1.0
    er_weight_dimension: float = 0.5
    er_weight_connections: float = 0.3
    er_weight_confidence: float = 0.2
    er_connections_for_full_score: int = 3

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
    "candidate_mult":    "KEYWORD_CANDIDATE_MULT",
    "candidate_min":     "KEYWORD_CANDIDATE_MIN",
    "rrf_k":             "KEYWORD_RRF_K",
    "name_alias_weight": "KEYWORD_NAME_ALIAS_WEIGHT",
    "column_weight":     "KEYWORD_COLUMN_WEIGHT",
    "name_ngram_weight": "NAME_NGRAM_WEIGHT",
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
    """Replace the process-wide config.

    For evaluation runs that need one setting to hold across a whole process.
    Per-call ``search(**overrides)`` reaches only the fused path; the diagnostic
    retriever views read this singleton, so a sweep has to set it here or the two
    disagree about candidate depth.
    """
    global _CONFIG
    _CONFIG = cfg


# ---------------------------------------------------------------------------
# Fusion primitives (pure; no I/O, no config)
# ---------------------------------------------------------------------------

_RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(s: str) -> str:
    """Lowercase and strip every non-alphanumeric character.

    Separators are removed, not replaced, so ``OM_FM.Amount_Fact``,
    ``om fm amount fact`` and ``omfmamountfact`` all collapse to one string —
    which is what lets :func:`promote_exact_match` recognise a query as a table
    name however the user punctuated it.

    Not the same as ``er_filter._normalize``, which only lowercases and trims:
    that one has to keep the separators, because it builds graph node keys.
    """
    return _RE_NON_ALNUM.sub("", s.lower())


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


def promote_exact_match(
    query: str,
    vector_ids: list[str],
    keyword_ids: list[str],
    table_names: list[str],
    table_aliases: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """
    If the normalized query matches a table name or alias exactly, move that
    table to rank 1 in both retrievers (inserting it if absent).

    Returns modified copies of (vector_ids, keyword_ids).
    """
    q_norm = _normalize(query)
    matched: Optional[str] = None

    for tname in table_names:
        if _normalize(tname) == q_norm:
            matched = tname
            break
        for alias in table_aliases.get(tname, []):
            if _normalize(str(alias)) == q_norm:
                matched = tname
                break
        if matched:
            break

    if matched is None:
        return vector_ids, keyword_ids

    def _to_front(lst: list[str], item: str) -> list[str]:
        return [item] + [x for x in lst if x != item]

    return _to_front(vector_ids, matched), _to_front(keyword_ids, matched)

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def candidate_depth(top_n: int, cfg: SearchConfig) -> int:
    """Per-retriever fetch depth.

    The ER pool is a floor on it only when expansion can actually happen:
    anchors come out of the caller's budget, so ``top_n <= er_anchor_n`` leaves
    no expansion slots and the deeper fetch would be pure waste.
    """
    er_active = cfg.er_candidates_n > 0 and top_n > cfg.er_anchor_n
    return max(top_n * cfg.candidate_mult, cfg.candidate_min,
               cfg.er_candidates_n if er_active else 0)


def dense_retrieve(collection, query: str, n: int, where: Optional[dict]) -> dict:
    """Dense retriever: ``{id: (meta, document, distance)}``, best first."""
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
) -> "tuple[list[str], dict[str, float], bool]":
    """Sparse retrievers: ``(ids, scores, available)``, or empty on failure.

    Imported here rather than at module scope: ``keyword_index`` imports the
    fusion primitives from this module, so a top-level import would cycle.
    """
    if cfg.keyword_weight <= 0.0:
        return [], {}, False

    from app.rag.keyword_index import get_keyword_index_service

    try:
        svc = get_keyword_index_service()
        # Build/load lazily so callers do not depend on loader-side warm-up.
        svc.warm(collection=collection, cfg=cfg)
        idx = svc.get()
        if idx is None:
            logger.debug("[table_info] Keyword index not yet built — vector-only.")
            return [], {}, False
        # Glossary-expanded query for the sparse retrievers only; the dense retriever and
        # the exact-match pin keep the original.
        hits = idx.search(expand_lexical_query(query), top_n=n, persona_id=persona_id, cfg=cfg)
        return [h[0] for h in hits], dict(hits), True
    except Exception as exc:
        logger.error("[table_info] Keyword search error (degrading to vector-only): %s", exc)
        return [], {}, False


def _combine_rankings(
    dense_ids: list[str],
    sparse_ids: list[str],
    sparse_scores: "dict[str, float]",
    sparse_available: bool,
    cfg: SearchConfig,
) -> "tuple[list[str], dict[str, float]]":
    """``(ranked_ids, score_by_id)`` — RRF when both ran, else whichever did."""
    if sparse_available and cfg.keyword_weight > 0.0 and cfg.vector_weight > 0.0:
        fused = fuse_rrf(dense_ids, sparse_ids, k=cfg.rrf_k,
                         w_a=cfg.vector_weight, w_b=cfg.keyword_weight)
        return [fid for fid, _ in fused], dict(fused)
    if cfg.keyword_weight == 0.0 or not sparse_available:
        return list(dense_ids), {}
    return list(sparse_ids), dict(sparse_scores)


def _add_related_tables(
    ranked_ids: list[str],
    top_n: int,
    cfg: SearchConfig,
) -> "tuple[list[str], set[str]]":
    """``(final_ids, related_ids)`` after the relationship-graph step."""
    if cfg.er_candidates_n <= 0:
        return ranked_ids[:top_n], set()

    n_anchors = min(cfg.er_anchor_n, top_n)
    anchors, related, used_graph = select_with_related_tables(
        ranked_ids[:cfg.er_candidates_n],
        n_anchors=n_anchors,
        n_related=max(0, min(cfg.er_related_n, top_n - n_anchors)),
        cfg=cfg,
    )
    # Only a real graph lookup earns the `er_expanded` label; the fallback takes
    # tables on text rank, and marking those would inflate the metric.
    return anchors + related, set(related) if used_graph else set()


def _fetch_missing_records(
    collection, ids: list[str], records: dict, where: Optional[dict],
) -> None:
    """Fetch metadata for ids the dense retriever never saw, in place.

    The ``where`` filter is passed on purpose: the keyword index filters against
    a persona snapshot frozen at build time, so an unfiltered fetch here would
    hand back a table whose persona tag has since changed.
    """
    missing = [tid for tid in ids if tid not in records]
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
            records[fid] = (meta, doc, _NO_DISTANCE)
    except Exception as exc:
        logger.warning("[table_info] Could not fetch kw-only metadata: %s", exc)


# Sentinel distance for a hit the dense retriever never returned
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
        ``fused_score``, ``match_source`` and ``er_expanded``.
    """
    cfg = (config or get_search_config()).with_overrides(**overrides)
    if not top_n or top_n <= 0:
        top_n = cfg.top_n

    # Startup runs the catalog load and the keyword-index build in the background so
    # the service comes up immediately; search is the one thing that must not run
    # against a half-loaded collection, so it waits here. No-op once ready, and a
    # no-op entirely in processes that never started the loader.
    # Imported here, not at module scope: `vectordb_loader` pulls in the loaders,
    # which reach back into this module.
    from app.rag.vectordb_loader import wait_until_ready
    wait_until_ready()

    collection = get_collection(
        COLLECTION_NAME, get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT"),
    )
    n_total = collection.count()
    if n_total == 0:
        logger.warning(
            "[table_info] Collection '%s' is empty — run the loader first.", COLLECTION_NAME,
        )
        return []

    where = persona_filter(persona_id)
    logger.info(
        '[table_info] Hybrid search: "%s…" (top_n=%d, persona_id=%s)',
        query[:60], top_n, persona_id,
    )

    candidate_n = candidate_depth(top_n, cfg)
    records = dense_retrieve(collection, query, min(candidate_n, n_total), where)
    dense_ids = list(records)

    sparse_ids, sparse_scores, sparse_available = _sparse_retrieve(
        collection, query, candidate_n, persona_id, cfg,
    )

    aliases = {tid: json.loads(meta.get("table_alias", "[]")) for tid, (meta, _, _) in records.items()}
    dense_ids, sparse_ids = promote_exact_match(query, dense_ids, sparse_ids, list(records), aliases)

    ranked_ids, fused_scores = _combine_rankings(dense_ids, sparse_ids, sparse_scores, sparse_available, cfg)
    final_ids, er_expanded = _add_related_tables(ranked_ids, top_n, cfg)

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
            "er_expanded":       tid in er_expanded,
        })
    return hits
