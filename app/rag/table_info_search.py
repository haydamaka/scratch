"""
Semantic search against the ChromaDB ``table_catalog`` collection populated by
``table_info_loader.TableInfoLoader``.

Public API:
    ``TableInfoSearch.get_search_hit_info(query, top_n, persona_id)``  hit dicts.
    ``TableInfoSearch.get_retriever_hits_detailed(...)``               per-retriever view.
    ``get_table_info_search()``                                        shared instance.

CLI:  python -m app.rag.table_info_search "query text"
"""

from __future__ import annotations

import os
import warnings

# Use pysqlite3 (sqlite >= 3.35) before any chromadb import.
if os.name == "posix":
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Silence noisy third-party startup warnings (must precede third-party imports).
from authlib.deprecate import AuthlibDeprecationWarning as _AuthlibDeprecationWarning
warnings.filterwarnings(action="ignore", category=_AuthlibDeprecationWarning)
warnings.filterwarnings(action="ignore", module="requests")

import sys
from typing import Optional

from app.core.logger import get_logger
from app.rag.chroma_db import (
    SEPARATOR_WIDTH,
    bootstrap_standalone,
)
from app.rag.vector_store import get_collection, persona_filter
from app.rag.embedding_vertex import get_vectordb_embedding_fn
from app.rag.hybrid_search import (
    COLLECTION_NAME,
    candidate_depth,
    get_search_config,
    search,
    dense_retrieve,
)
from app.rag.keyword_index import get_keyword_index_service
from app.rag.query_prep import expand_lexical_query

logger = get_logger(__name__)

DEFAULT_TOP_N   = 5   # signature default; `SearchConfig.top_n` is authoritative


def _normalized_name(name: str) -> str:
    """Lower-cased, unquoted table name — how requested names are matched."""
    return str(name).strip().strip('"').strip("`").lower()


def _name_lookup_map(collection) -> "dict[str, list[str]]":
    """``{normalized name: [actual names]}`` for the whole catalog.

    Requested names arrive from eval files and callers with assorted case and
    quoting, so they are resolved through this rather than used directly.
    """
    lookup: "dict[str, list[str]]" = {}
    try:
        catalog = collection.get(include=["metadatas"])
        for meta in catalog.get("metadatas") or []:
            table_name = str((meta or {}).get("table_name") or "")
            if table_name:
                lookup.setdefault(_normalized_name(table_name), []).append(table_name)
    except Exception as exc:
        logger.warning("[table_info] Could not build the table-name map: %s", exc)
    return lookup


def _name_where(table_name: str, persona_id: Optional[int]) -> dict:
    """Chroma filter for one table name, persona-scoped when asked."""
    clause = {"table_name": {"$eq": table_name}}
    persona_clause = persona_filter(persona_id)
    return {"$and": [clause, persona_clause]} if persona_clause else clause



class TableInfoSearch:
    """Catalog-facing search surface.

    The pipeline itself lives in :mod:`app.rag.hybrid_search`; what remains here
    is the ChromaDB-facing lookups (distances and records by table name) and the
    per-retriever diagnostic views the eval harness reports.
    """

    def get_search_hit_info(
        self,
        query: str,
        top_n: int = DEFAULT_TOP_N,
        persona_id: Optional[int] = None,
        **overrides,
    ) -> list[dict]:
        """Run hybrid (vector + lexical) search against the table catalog.

        Thin wrapper over :func:`hybrid_search.search`, which owns the pipeline
        and every knob; ``overrides`` are ``SearchConfig`` fields for this call.
        Kept under this name because the API endpoint, the CLI and both eval
        harnesses call it.
        """
        return search(query, top_n=top_n, persona_id=persona_id, **overrides)

    # —— diagnostic retriever views ——————————————————————————————————————

    def _retriever_hits(
        self,
        query: str,
        top_n: int,
        persona_id: Optional[int],
    ) -> "tuple[list[dict], list[dict], list[dict]]":
        """``(vector, keyword, name_alias)`` hits, each ranked within its retriever.

        One embed and one lexical breakdown feed all three lists.

        Depth here deliberately drops the ER pool floor (``er_candidates_n=0``): the
        fused path fetches deeper only so ``er_filter`` has a tail to read below
        the head. These lists are therefore the retrievers at their own depth, not
        the lists the fused path ranked.
        """
        cfg = get_search_config()
        if not top_n or top_n <= 0:
            top_n = cfg.top_n

        collection = get_collection(
            COLLECTION_NAME, get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT"),
        )
        n_total = collection.count()
        if n_total == 0:
            return [], [], []

        candidate_n = candidate_depth(top_n, cfg.with_overrides(er_candidates_n=0))
        found = dense_retrieve(
            collection, query, min(candidate_n, n_total), persona_filter(persona_id),
        )
        vector_hits = [
            {
                "table_name":        meta["table_name"],
                "table_description": meta.get("table_description", ""),
                "distance":          round(distance, 4),
                "rank":              rank,
            }
            for rank, (meta, _, distance) in enumerate(found.values(), start=1)
        ]

        breakdown: dict = {}
        if cfg.keyword_weight > 0.0:
            try:
                svc = get_keyword_index_service()
                svc.warm(collection=collection, cfg=cfg)
                idx = svc.get()
                if idx is not None:
                    # The same expansion the fused path applies, so this view
                    # reports the retriever as it actually runs.
                    breakdown = idx.search_with_breakdown(
                        expand_lexical_query(query),
                        top_n=candidate_n,
                        persona_id=persona_id,
                        cfg=cfg,
                    )
            except Exception as exc:
                logger.error("[table_info] retriever breakdown error: %s", exc)

        described = {h["table_name"]: h["table_description"] for h in vector_hits}
        return (
            vector_hits,
            self._format_hits(collection, breakdown.get("keyword_hits", []), described),
            self._format_hits(collection, breakdown.get("name_alias_hits", []), described),
        )

    @staticmethod
    def _format_hits(collection, scored: list, described: dict) -> list[dict]:
        """Format ``(id, score)`` pairs, back-filling descriptions one id at a time."""
        hits: list[dict] = []
        for rank, (table_id, score) in enumerate(scored, start=1):
            description = described.get(table_id, "")
            if not description:
                try:
                    fetched = collection.get(ids=[table_id], include=["metadatas"])
                    if fetched.get("metadatas"):
                        description = fetched["metadatas"][0].get("table_description", "")
                except Exception:
                    # Cosmetic in a diagnostic view — never drop the hit for it.
                    pass
            hits.append({
                "table_name":        table_id,
                "table_description": description,
                "keyword_score":     round(score, 6),
                "rank":              rank,
            })
        return hits

    def get_retriever_hits_detailed(
        self,
        query: str,
        top_n: int = DEFAULT_TOP_N,
        persona_id: Optional[int] = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Return vector hits, combined keyword hits, and name+alias keyword hits."""
        return self._retriever_hits(query, top_n, persona_id)

    def get_query_distances_for_tables(
        self,
        query: str,
        table_names: list[str],
        persona_id: Optional[int] = None,
    ) -> dict[str, Optional[float]]:
        """Return vector distance from ``query`` to each requested table name.

        Distances are fetched by querying Chroma with a strict table_name filter,
        so this works even when a table is not in top-N vector results.
        """
        out: dict[str, Optional[float]] = {t: None for t in table_names}
        if not table_names:
            return out

        collection = get_collection(
            COLLECTION_NAME,
            get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT"),
        )

        n_total = collection.count()
        if n_total == 0:
            return out

        query_embedding = get_vectordb_embedding_fn(task="RETRIEVAL_QUERY")([query])[0]

        lookup = _name_lookup_map(collection)

        for table_name in table_names:
            candidates = lookup.get(_normalized_name(table_name)) or [table_name]
            best: Optional[float] = None

            for candidate in candidates:
                try:
                    res = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=1,
                        include=["distances"],
                        where=_name_where(candidate, persona_id),
                    )
                    dists = res.get("distances") or []
                    first = dists[0] if dists else []
                    if first:
                        dist = round(float(first[0]), 4)
                        best = dist if best is None else min(best, dist)
                except Exception as exc:
                    logger.warning(
                        "[table_info] Could not compute distance for '%s': %s",
                        candidate,
                        exc,
                    )

            out[table_name] = best

        return out

    def get_table_records(
        self,
        table_names: list[str],
        persona_id: Optional[int] = None,
    ) -> dict[str, dict]:
        """Return full metadata+document for each requested table name.

        Output shape:
            {
              "<requested_name>": {
                "found": bool,
                "resolved_table_name": str,
                "metadata": dict,
                "document": str,
              }
            }
        """
        out: dict[str, dict] = {
            t: {
                "found": False,
                "resolved_table_name": "",
                "metadata": {},
                "document": "",
            }
            for t in table_names
        }
        if not table_names:
            return out

        collection = get_collection(
            COLLECTION_NAME,
            get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT"),
        )
        if collection.count() == 0:
            return out

        lookup = _name_lookup_map(collection)

        for requested in table_names:
            candidates = lookup.get(_normalized_name(requested)) or [requested]

            for candidate in candidates:
                try:
                    res = collection.get(
                        where=_name_where(candidate, persona_id),
                        include=["metadatas", "documents"],
                    )
                    metas = res.get("metadatas") or []
                    docs = res.get("documents") or []
                    if metas:
                        out[requested] = {
                            "found": True,
                            "resolved_table_name": candidate,
                            "metadata": metas[0] or {},
                            "document": docs[0] if docs else "",
                        }
                        break
                except Exception as exc:
                    logger.warning("[table_info] Could not fetch record for '%s': %s", candidate, exc)

        return out

def print_hit_metadata(hit: dict) -> None:
    """Print the stored ChromaDB metadata fields for one search hit."""
    table_name  = hit.get("table_name", "")
    description = hit.get("table_description", "")
    domain      = hit.get("domain_mapping", "") or "(none)"
    persona_id  = hit.get("persona_id", -1)
    alias_list  = hit.get("table_alias", [])
    distance    = hit.get("distance", 0.0)

    logger.debug(
        "[table_info] print_hit_metadata: table='%s' dist=%.4f", table_name, distance
    )

    desc_preview = (
        description[:100] + "..." if len(description) > 100 else description
    )
    print(f"  table_name   : {table_name}")
    print(f"  description  : {desc_preview}")
    print(f"  domain       : {domain}")
    print(f"  aliases      : {', '.join(str(a) for a in alias_list) or '(none)'}")
    print(f"  persona_id   : {persona_id if persona_id != -1 else '(none)'}")
    print(f"  distance     : {distance:.4f}")


def print_hit_document(hit: dict, max_lines: int = 15) -> None:
    """Print the embedded document text for one search hit (truncated)."""
    doc = hit.get("document", "")
    if not doc:
        logger.warning(
            "[table_info] print_hit_document: no document for '%s'.",
            hit.get("table_name", "?"),
        )
        print("  (no document text stored)")
        return

    lines       = doc.splitlines()
    total_lines = len(lines)
    display     = lines[:max_lines]

    logger.debug(
        "[table_info] print_hit_document: table='%s', %d chars, %d lines (showing %d).",
        hit.get("table_name", ""), len(doc), total_lines, len(display),
    )

    print(f"  Document  ({len(doc)} chars / {total_lines} lines):")
    for line in display:
        print(f"    {line}")
    if total_lines > max_lines:
        print(f"    ... ({total_lines - max_lines} more lines not shown)")


def print_search_results(
    query: str,
    hits: list[dict],
    *,
    show_document: bool = True,
    document_max_lines: int = 15,
) -> None:
    """Print all search results using the dedicated per-hit printers."""
    logger.info(
        '[table_info] print_search_results: query="%s", hits=%d, show_doc=%s',
        query[:60], len(hits), show_document,
    )

    print(f'\nSearch query: "{query}"')
    print("=" * SEPARATOR_WIDTH)

    if not hits:
        logger.warning('[table_info] No results for query: "%s"', query)
        print("  (no results — is the collection loaded?)")
        return

    for rank, hit in enumerate(hits, start=1):
        print(f"\n[{rank}]  {hit.get('table_name', '?')}")
        print("-" * SEPARATOR_WIDTH)

        print("  [metadata]")
        print_hit_metadata(hit)

        if show_document:
            print("  [document]")
            print_hit_document(hit, max_lines=document_max_lines)

    print()
    logger.info("[table_info] Displayed %d result(s).", len(hits))


_default_instance: "TableInfoSearch | None" = None


def get_table_info_search() -> TableInfoSearch:
    """Return the module-level ``TableInfoSearch`` singleton."""
    global _default_instance
    if _default_instance is None:
        _default_instance = TableInfoSearch()
    return _default_instance


def main() -> None:
    """
    Entry point when the module is run as a standalone script.

    Uses the arguments as the search query.

    Example::

        python -m app.rag.table_info_search "credit facility agreement outstanding"
    """
    bootstrap_standalone()

    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print('Usage: python -m app.rag.table_info_search "query text"')
        return

    query = " ".join(sys.argv[1:])
    logger.info('[table_info] CLI search mode: query="%s"', query)
    hits = get_table_info_search().get_search_hit_info(query)
    print_search_results(query, hits)


if __name__ == "__main__":
    main()
