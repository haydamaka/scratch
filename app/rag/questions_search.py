"""
Runs semantic search against the ChromaDB ``question_catalog`` collection that
is populated by ``questions_loader.QuestionInfoLoader``.

Public API:
    ``QuestionInfoSearch.get_search_hit_info(query, top_n)``
        Search hit dicts (question, ground_truth_sql, expected_table,
        table_names, domains, distance, document).
    ``QuestionInfoSearch.search_questions(query, top_n)``
        Compact format (question, ground_truth_sql, expected_table).

Module-level convenience:
    ``search_questions(query, top_n)``    → compact format
    ``get_question_info_search()``        → shared singleton

Asymmetric embedding:
    Index time  → task="RETRIEVAL_DOCUMENT"  (see questions_loader.py)
    Query time  → task="RETRIEVAL_QUERY"     (pre-computed and passed as
                   ``query_embeddings`` so ChromaDB never re-embeds the query)

Usage (CLI):
    python -m app.rag.questions_search "query text"
"""

from __future__ import annotations

import os
import warnings

# Use pysqlite3 (sqlite >= 3.35) before any chromadb import.
if os.name == "posix":
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ── Silence noisy third-party startup warnings ──────────────────────────────
# Must come before ANY third-party import (see table_info_loader.py for details).
from authlib.deprecate import AuthlibDeprecationWarning as _AuthlibDeprecationWarning
warnings.filterwarnings("ignore", category=_AuthlibDeprecationWarning)
warnings.filterwarnings("ignore", module="requests")
# ────────────────────────────────────────────────────────────────────────────

import json
import sys

from app.core.logger import get_logger
from app.rag.chroma_db import (
    SEPARATOR_WIDTH,
    bootstrap_standalone,
)
from app.rag.vector_store import get_collection, persona_filter
from app.rag.embedding_vertex import get_vectordb_embedding_fn

logger = get_logger(__name__)

# ────────────────────────────────────────────────
# Module-level constants
# ────────────────────────────────────────────────
COLLECTION_NAME   = "question_catalog"
DEFAULT_TOP_N     = 3

# ────────────────────────────────────────────────
# Main class
# ────────────────────────────────────────────────

class QuestionInfoSearch:

    def get_search_hit_info(
        self,
        query: str,
        top_n: int = DEFAULT_TOP_N,
        persona_id: int | None = None,
    ) -> list[dict]:
        if not top_n or top_n <= 0:
            top_n = DEFAULT_TOP_N

        collection = get_collection(
            COLLECTION_NAME,
            get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT"),
        )

        query_embedding = get_vectordb_embedding_fn(task="RETRIEVAL_QUERY")([query])[0]

        n_results = min(top_n, collection.count())
        if n_results == 0:
            logger.warning(
                "[questions] Collection '%s' is empty — run the loader first "
                "(python -m app.rag.questions_loader).",
                COLLECTION_NAME,
            )
            return []

        where_filter = persona_filter(persona_id)

        logger.info(
            '[questions] Searching: "%s…" (top_n=%d, persona_id=%s)',
            query[:60], top_n, persona_id,
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            **({"where": where_filter} if where_filter else {}),
        )

        hits: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "question":         meta["question"],
                "ground_truth_sql": meta.get("ground_truth_sql", ""),
                "expected_table":   json.loads(meta.get("expected_table", "[]")),
                "table_names":      meta.get("table_names", ""),
                "domains":          meta.get("domains", ""),
                "persona_ids":      json.loads(meta.get("persona_ids", "[]")),
                "distance":         round(dist, 4),
                "document":         doc,
            })

        return hits

    def search_questions(
        self,
        query: str,
        top_n: int = DEFAULT_TOP_N,
        persona_id: int | None = None,
    ) -> list[dict]:
        return [
            {
                "question":         h["question"],
                "ground_truth_sql": h["ground_truth_sql"],
                "expected_table":   h["expected_table"],
                "persona_ids":      h.get("persona_ids", []),
            }
            for h in self.get_search_hit_info(
                query,
                top_n=top_n,
                persona_id=persona_id,
            )
        ]


# ────────────────────────────────────────────────
# Dedicated result-printing functions
# ────────────────────────────────────────────────

def print_hit_metadata(hit: dict) -> None:
    """Print the stored ChromaDB metadata fields for one search hit."""
    question    = hit.get("question", "")
    sql         = hit.get("ground_truth_sql", "") or "(none)"
    tables      = hit.get("table_names", "") or "(none)"
    domains     = hit.get("domains", "") or "(none)"
    persona_ids = hit.get("persona_ids", [])
    distance    = hit.get("distance", 0.0)

    logger.debug(
        "[questions] print_hit_metadata: dist=%.4f tables='%s'", distance, tables
    )

    q_preview   = question[:100] + "..." if len(question) > 100 else question
    sql_preview = sql[:120] + "..." if len(sql) > 120 else sql
    print(f"  question    : {q_preview}")
    print(f"  tables      : {tables}")
    print(f"  domains     : {domains}")
    print(f"  persona_ids : {persona_ids or '(none)'}")
    print(f"  ground_sql  : {sql_preview}")
    print(f"  distance    : {distance:.4f}")


def print_hit_document(hit: dict, max_lines: int = 15) -> None:
    """Print the embedded document text (the question) for one search hit."""
    doc = hit.get("document", "")
    if not doc:
        logger.warning(
            "[questions] print_hit_document: no document for hit."
        )
        print("  (no document text stored)")
        return

    lines       = doc.splitlines()
    total_lines = len(lines)
    display     = lines[:max_lines]

    logger.debug(
        "[questions] print_hit_document: %d chars, %d lines (showing %d).",
        len(doc), total_lines, len(display),
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
        '[questions] print_search_results: query="%s", hits=%d, show_doc=%s',
        query[:60], len(hits), show_document,
    )

    print(f'\nSearch query: "{query}"')
    print("=" * SEPARATOR_WIDTH)

    if not hits:
        logger.warning('[questions] No results for query: "%s"', query)
        print("  (no results — is the collection loaded?)")
        return

    for rank, hit in enumerate(hits, start=1):
        q_title = hit.get("question", "?")
        q_title = q_title[:60] + "..." if len(q_title) > 60 else q_title
        print(f"\n[{rank}]  {q_title}")
        print("-" * SEPARATOR_WIDTH)

        print("  [metadata]")
        print_hit_metadata(hit)

        if show_document:
            print("  [document]")
            print_hit_document(hit, max_lines=document_max_lines)

        print()
    logger.info("[questions] Displayed %d result(s).", len(hits))


# ────────────────────────────────────────────────
# Module-level convenience helpers
# ────────────────────────────────────────────────

_default_instance: "QuestionInfoSearch | None" = None


def get_question_info_search() -> QuestionInfoSearch:
    global _default_instance
    if _default_instance is None:
        _default_instance = QuestionInfoSearch()
    return _default_instance


def search_questions(
    query: str,
    top_n: int = DEFAULT_TOP_N,
    persona_id: int | None = None,
) -> list[dict]:
    return get_question_info_search().search_questions(
        query,
        top_n=top_n,
        persona_id=persona_id,
    )

# ────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────

def main() -> None:
    """
    Example::
        python -m app.rag.questions_search "active committed facilities held for investment"
    """
    bootstrap_standalone()

    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print('Usage: python -m app.rag.questions_search "query text"')
        return

    query = " ".join(sys.argv[1:])
    logger.info('[questions] CLI search mode: query="%s"', query)
    hits = get_question_info_search().get_search_hit_info(query)
    print_search_results(query, hits)


if __name__ == "__main__":
    main()
