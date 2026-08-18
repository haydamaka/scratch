"""
Loads table metadata from the persona API into the ChromaDB ``table_catalog``
collection for semantic table pre-filtering.

Public API (``TableInfoLoader``):
    ``upsert_one(item)``      Upsert a single table record. Idempotent.
    ``load(retriever)``       Async. Bulk-upsert everything a retriever returns.
    ``load_from_api()``       Async. Shorthand for the service retriever.
    ``load_from_json(path)``  Blocking. Shorthand for a file retriever.

CLI:  python -m app.rag.table_info_loader
"""

from __future__ import annotations

import os

import app.rag.sqlite_shim  # noqa: F401  — must precede any chromadb import

import warnings

# Silence noisy third-party startup warnings (must precede third-party imports).
from authlib.deprecate import AuthlibDeprecationWarning as _AuthlibDeprecationWarning
warnings.filterwarnings("ignore", category=_AuthlibDeprecationWarning)
warnings.filterwarnings("ignore", module="requests")

import asyncio
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

from app.core.logger import get_logger
from app.rag.chroma_db import (
    UPDATE_EXISTING,
    bootstrap_standalone,
)
from app.rag.vector_store import (
    VectorCollection,
    get_or_create_collection,
    store_location,
)
from app.rag.embedding_vertex import (
    get_vectordb_embedding_fn,
    DOCUMENT_MAX_CHARS,
    upsert_batch_size,
)

logger = get_logger(__name__)

COLLECTION_NAME = "table_catalog"

class TableInfoLoader:
    """Builds and maintains the ChromaDB ``table_catalog`` collection."""

    # Number of records actually changed (new or differing) by the last load.
    last_changed_count: int = 0

    @staticmethod
    def _token_estimate(text: str) -> int:
        # Cheap estimate: words + punctuation tokens; good enough for truncation.
        return len(re.findall(r"[A-Za-z0-9_]+|[^\s]", text or ""))

    @staticmethod
    def _build_compact_document(item: dict) -> str:
        """Fallback document used when full table payload repeatedly fails embedding."""
        table_name = str(item.get("tableName") or "")
        desc = str(item.get("tableDescription") or "")
        doc = f"TABLE: {table_name}\nDESCRIPTION: {desc}"
        return doc[: min(DOCUMENT_MAX_CHARS, 1200)]

    def _try_compact_upsert(self, collection: VectorCollection, doc_id: str, item: dict, meta: dict, source: str) -> bool:
        compact_doc = self._build_compact_document(item)
        try:
            collection.upsert(ids=[doc_id], documents=[compact_doc], metadatas=[meta])
            logger.warning(
                "[table_info] compact fallback succeeded for '%s' (%s). full_doc failed; compact chars=%d token_est=%d",
                doc_id,
                source,
                len(compact_doc),
                self._token_estimate(compact_doc),
            )
            return True
        except Exception as retry_exc:
            self._log_extended_error(f"Compact fallback failed in {source}", doc_id, item, compact_doc, meta, retry_exc)
            return False

    def _get_or_create_collection(self) -> VectorCollection:
        logger.info(
            "[table_info] opening/creating vector collection '%s' at '%s' ...",
            COLLECTION_NAME, store_location(),
        )
        embed_fn = get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT")
        collection = get_or_create_collection(COLLECTION_NAME, embed_fn)
        logger.info("[table_info] vector collection '%s' ready.", COLLECTION_NAME)
        return collection

    @staticmethod
    def _normalize_persona_ids(raw_persona_ids) -> list[int]:
        """Normalize persona IDs because upstream payloads may send int/str/list with duplicates."""
        if isinstance(raw_persona_ids, int):
            values = [raw_persona_ids]
        elif isinstance(raw_persona_ids, str):
            values = [p for p in re.split(r"[\s,]+", raw_persona_ids.strip()) if p]
        elif isinstance(raw_persona_ids, list):
            values = raw_persona_ids
        else:
            return []

        normalized: list[int] = []
        seen: set[int] = set()
        for value in values:
            try:
                pid = int(value)
            except (TypeError, ValueError):
                continue
            if pid in seen:
                continue
            seen.add(pid)
            normalized.append(pid)
        return normalized

    def _persona_ids_of(self, item: dict) -> list[int]:
        return self._normalize_persona_ids(item.get("personaId"))

    @staticmethod
    def _primary_persona_id(persona_ids: list[int]) -> int:
        return persona_ids[0] if persona_ids else -1

    def _partition_by_existence(
        self,
        collection: VectorCollection,
        items: list[dict],
    ) -> "tuple[list[dict], list[str]]":
        """
        Split ``items`` into ``(to_upsert, skipped_ids)`` according to
        ``UPDATE_EXISTING``
        """
        items = list(items)
        if UPDATE_EXISTING or not items:
            return items, []

        fetched = collection.get(
            ids=[it["tableName"] for it in items],
            include=["metadatas"],
        )
        stored_persona_ids: dict[str, list[int]] = {
            _id: json.loads((meta or {}).get("persona_ids", "[]"))
            for _id, meta in zip(
                fetched.get("ids") or [],
                fetched.get("metadatas") or [],
            )
        }

        to_upsert: list[dict] = []
        skipped_ids: list[str] = []
        for it in items:
            table_name = it["tableName"]
            persona_ids = self._persona_ids_of(it)
            if table_name in stored_persona_ids and stored_persona_ids[table_name] == persona_ids:
                skipped_ids.append(table_name)
            else:
                to_upsert.append(it)
        return to_upsert, skipped_ids

    def _build_document(self, item: dict) -> str:
        """
        Build a plain-text string for embedding from one table record.
          - Table name
          - Aliases (if present)
          - Description

        Column names are intentionally excluded from the dense (semantic) embedding payload
        and stored in metadata instead.
        """
        parts: list[str] = []

        parts.append(f"TABLE: {item['tableName']}")

        aliases = item.get("tableAlias") or []
        if aliases:
            parts.append(f"ALIASES: {', '.join(str(a) for a in aliases)}")

        parts.append(f"DESCRIPTION: {item.get('tableDescription') or ''}")

        header = "\n".join(parts)

        # Keep the dense (semantic) embedding focused on table-level semantics only.
        return header[: DOCUMENT_MAX_CHARS]

    @staticmethod
    def _coerce_metadata_scalar(value):
        """Chroma metadata accepts only str/int/float/bool scalar values."""
        if isinstance(value, (str, int, float, bool)):
            return value
        if value is None:
            return ""
        if isinstance(value, (list, dict, tuple, set)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    @classmethod
    def _sanitize_metadata(cls, metadata: dict) -> dict:
        return {k: cls._coerce_metadata_scalar(v) for k, v in metadata.items()}

    @staticmethod
    def _existing_table_ids(collection: VectorCollection) -> set[str]:
        """Return all currently indexed table ids from the vector collection."""
        try:
            fetched = collection.get(include=["metadatas"])
            return {str(_id) for _id in (fetched.get("ids") or [])}
        except Exception as exc:
            logger.warning(
                "[table_info] Could not fetch existing table ids; continuing without pre-filter: %s",
                exc,
            )
            return set()

    @staticmethod
    def _filter_missing_records(items: list[dict], existing_ids: set[str]) -> tuple[list[dict], int]:
        filtered = [it for it in items if str(it.get("tableName") or "") not in existing_ids]
        return filtered, len(items) - len(filtered)

    def _build_metadata(self, item: dict) -> dict:
        """
        Produce a flat ChromaDB-safe metadata dict from one table record.
        """
        persona_ids = self._persona_ids_of(item)
        raw_cols = item.get("columns") or []
        col_names: list[str] = []
        for c in raw_cols:
            if isinstance(c, str):
                if c:
                    col_names.append(c)
            elif isinstance(c, dict):
                name = c.get("columnName")
                if name:
                    col_names.append(str(name))

        metadata = {
            "table_name":        item["tableName"],
            "table_description": item.get("tableDescription") or "",
            "version":           item.get("version") or 0,
            "table_specific_rules": item.get("tableSpecificRules") or "",
            "domain_mapping":    item.get("domainMapping") or "",
            "persona_name":      item.get("personaName") or "",
            "persona_id": self._primary_persona_id(persona_ids),
            "persona_ids": json.dumps(persona_ids),
            "table_alias": json.dumps(item.get("tableAlias") or []),
            "column_names": json.dumps(col_names, ensure_ascii=False),
            "column_count": len(col_names),
        }
        for pid in persona_ids:
            metadata[f"persona_id_{pid}"] = 1
        return self._sanitize_metadata(metadata)

    def upsert_one(self, item: dict) -> None:
        table_name = item["tableName"]
        persona_id = self._primary_persona_id(self._persona_ids_of(item))

        collection = self._get_or_create_collection()

        if not UPDATE_EXISTING:
            _, skipped_ids = self._partition_by_existence(collection, [item])
            if skipped_ids:
                logger.info(
                    "[table_info] upsert_one: '%s' (persona_id=%s) already exists — "
                    "skipping (update_existing=False).",
                    table_name, persona_id,
                )
                return

        logger.info("[table_info] upsert_one: building document for '%s' ...", table_name)

        document = self._build_document(item)
        metadata = self._build_metadata(item)

        logger.debug(
            "[table_info] upsert_one '%s': document=%d chars, columns=%d, "
            "persona_id=%s, domain='%s'",
            table_name,
            len(document),
            len(item.get("columns") or []),
            metadata["persona_id"],
            metadata["domain_mapping"] or "(none)",
        )

        count_before = collection.count()

        collection.upsert(
            ids       = [table_name],
            documents = [document],
            metadatas = [metadata],
        )

        count_after = collection.count()
        action = "inserted" if count_after > count_before else "updated"
        logger.info(
            "[table_info] upsert_one: '%s' %s → collection '%s' "
            "(collection size: %d → %d).",
            table_name, action, COLLECTION_NAME, count_before, count_after,
        )

    def load_from_json(
        self,
        json_path: "str | Path | None" = None,
        *,
        max_records: Optional[int] = None,
        show_progress: bool = True,
    ) -> VectorCollection:
        """Load from a file. Synchronous for callers that are not in an event
        loop; inside one, await ``load(LocalTableInfoRetriever(path))`` instead."""
        from app.rag.local_table_info_retriever import LocalTableInfoRetriever

        return asyncio.run(self.load(
            LocalTableInfoRetriever(json_path),
            max_records=max_records,
            show_progress=show_progress,
        ))

    async def load(
        self,
        retriever,
        *,
        max_records: Optional[int] = None,
        show_progress: bool = True,
    ) -> VectorCollection:
        """Load whatever the retriever hands over. The only difference between a
        service load and a file load is which retriever arrives here."""
        logger.info("[table_info] load: reading from %s ...", retriever.source)
        table_records = await retriever.fetch()

        total_in_source = len(table_records)
        logger.info("[table_info] load: received %d records.", total_in_source)

        if max_records is not None:
            table_records = table_records[:max_records]
            logger.info(
                "[table_info] max_records=%d → processing %d of %d records.",
                max_records, len(table_records), total_in_source,
            )

        collection   = self._get_or_create_collection()
        count_before = collection.count()
        logger.info(
            "[table_info] Collection '%s' current size: %d documents.",
            COLLECTION_NAME, count_before,
        )

        # Load-only-missing strategy: fetch existing ids once, then keep only new tables.
        existing_ids = self._existing_table_ids(collection)
        source_before_filter = len(table_records)
        table_records, skipped_existing = self._filter_missing_records(table_records, existing_ids)
        if existing_ids:
            logger.info(
                "[table_info] pre-filter existing ids: %d source -> %d missing (%d already indexed).",
                source_before_filter,
                len(table_records),
                skipped_existing,
            )

        total        = len(table_records)
        upserted     = 0
        skipped_total = skipped_existing
        changed_total = 0

        batch_size = upsert_batch_size()
        for batch_start in range(0, total, batch_size):
            batch = table_records[batch_start : batch_start + batch_size]

            if not UPDATE_EXISTING:
                batch, skipped_ids = self._partition_by_existence(collection, batch)
                if skipped_ids:
                    skipped_total += len(skipped_ids)
                    logger.info(
                        "[table_info] batch @%d: skipping %d existing record(s) "
                        "(update_existing=False).",
                        batch_start + 1, len(skipped_ids),
                    )
                if not batch:
                    continue

            ids, documents, metadatas = [], [], []
            for item in batch:
                doc  = self._build_document(item)
                meta = self._build_metadata(item)
                ids.append(item["tableName"])
                documents.append(doc)
                metadatas.append(meta)
                logger.debug(
                    "[table_info]   '%s': %d cols, doc=%d chars, persona_id=%s",
                    item["tableName"],
                    len(item.get("columns") or []),
                    len(doc),
                    meta["persona_id"],
                )

            changed_ids = self._changed_id_set(collection, ids, documents, metadatas)

            try:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                upserted += len(batch)
                changed_total += len(changed_ids)
            except Exception as batch_exc:
                logger.error(
                    "[table_info] batch [%d-%d] FAILED during API load (%s). "
                    "Retrying %d docs one-by-one ...",
                    batch_start + 1, batch_start + len(batch), batch_exc, len(batch),
                )
                n_ok = n_fail = 0
                for item, doc_id, doc, meta in zip(batch, ids, documents, metadatas):
                    try:
                        collection.upsert(ids=[doc_id], documents=[doc], metadatas=[meta])
                        n_ok += 1
                        upserted += 1
                        if doc_id in changed_ids:
                            changed_total += 1
                    except Exception as exc:
                        n_fail += 1
                        skipped_total += 1
                        if self._is_vertex_400(exc):
                            self._log_extended_error("Vertex 400 during API load", doc_id, item, doc, meta, exc)
                            if self._try_compact_upsert(collection, doc_id, item, meta, "API load"):
                                n_ok += 1
                                n_fail -= 1
                                upserted += 1
                                skipped_total -= 1
                                if doc_id in changed_ids:
                                    changed_total += 1
                        elif self._is_metadata_scalar_error(exc):
                            self._log_extended_error("Metadata scalar validation error during API load", doc_id, item, doc, meta, exc)
                        else:
                            logger.error(
                                "[table_info] SKIP '%s' during API load: exc_type=%s exc=%s",
                                doc_id,
                                type(exc).__name__,
                                exc,
                            )
                logger.info("[table_info] batch recovery: %d ok, %d skipped.", n_ok, n_fail)

            if show_progress:
                logger.info(
                    "[table_info] batch [%d-%d]: %d/%d upserted.",
                    batch_start + 1, batch_start + len(batch), upserted, total,
                )

        count_after = collection.count()
        logger.info(
            "[table_info] load: done. upserted=%d, skipped=%d, "
            "collection size: %d → %d (%+d). Store: %s",
            upserted, skipped_total, count_before, count_after,
            count_after - count_before, store_location(),
        )
        self.last_changed_count = changed_total
        logger.info("[table_info] load: %d record(s) actually changed.", changed_total)
        return collection

    async def load_from_api(
        self,
        *,
        max_records: Optional[int] = None,
        show_progress: bool = True,
    ) -> VectorCollection:
        """Load from the training-intelligence service."""
        from app.rag.table_info_retriever import TableInfoRetriever

        return await self.load(
            TableInfoRetriever(), max_records=max_records, show_progress=show_progress,
        )

    @staticmethod
    # Only ever finds anything under Milvus. Chroma's pre-filter above already
    # dropped every id the collection holds, so the diff sees an empty fetch and
    # reports "all changed"; milvus_db.get() ignores the pre-filter, so there the
    # comparison is real.
    def _changed_id_set(
        collection: VectorCollection,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> set[str]:
        """Return ids that are new or differ from what is already stored."""
        if not ids:
            return set()
        try:
            fetched = collection.get(ids=ids, include=["documents", "metadatas"])
        except Exception:
            return set(ids)

        existing_docs = {
            _id: doc
            for _id, doc in zip(fetched.get("ids") or [], fetched.get("documents") or [])
        }
        existing_meta = {
            _id: meta
            for _id, meta in zip(fetched.get("ids") or [], fetched.get("metadatas") or [])
        }

        changed: set[str] = set()
        for _id, doc, meta in zip(ids, documents, metadatas):
            if _id not in existing_docs or existing_docs.get(_id) != doc or existing_meta.get(_id) != meta:
                changed.add(_id)
        return changed

    @staticmethod
    def _is_vertex_400(exc: Exception) -> bool:
        msg = str(exc)
        return "Error: 400 POST" in msg and "text-embedding-005" in msg

    @staticmethod
    def _is_metadata_scalar_error(exc: Exception) -> bool:
        return "Expected metadata value to be a str, int, float or bool" in str(exc)

    @staticmethod
    def _metadata_type_profile(meta: dict) -> dict[str, str]:
        return {k: type(v).__name__ for k, v in (meta or {}).items()}

    @staticmethod
    def _log_extended_error(kind: str, doc_id: str, item: dict, doc: str, meta: dict, exc: Exception) -> None:
        logger.error(
            "[table_info] %s for '%s': exc_type=%s exc_repr=%r context=%s meta_types=%s",
            kind,
            doc_id,
            type(exc).__name__,
            exc,
            TableInfoLoader._error_context(item, doc, meta),
            TableInfoLoader._metadata_type_profile(meta),
        )
        logger.debug("[table_info] %s traceback for '%s':\n%s", kind, doc_id, traceback.format_exc())

    @staticmethod
    def _error_context(item: dict, doc: str, meta: dict) -> dict:
        """Compact payload for debugging record-specific embedding failures."""
        control_chars = sum(1 for ch in doc if ord(ch) < 32 and ch not in ("\n", "\r", "\t"))
        non_ascii = sum(1 for ch in doc if ord(ch) > 127)
        return {
            "table": str(item.get("tableName") or ""),
            "doc_chars": len(doc),
            "doc_sha1_12": hashlib.sha1(doc.encode("utf-8", errors="replace")).hexdigest()[:12],
            "desc_chars": len(str(item.get("tableDescription") or "")),
            "columns": len(item.get("columns") or []),
            "persona_id": meta.get("persona_id"),
            "control_chars": control_chars,
            "non_ascii_chars": non_ascii,
            "doc_head": doc[:220],
            "doc_tail": doc[-220:],
        }


def main() -> None:
    bootstrap_standalone()

    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logger.info("[table_info] CLI load mode: loading from API.")
    loader = TableInfoLoader()
    asyncio.run(loader.load_from_api())

    # A full CLI load is a real bulk load → record the DB version/metadata
    # (ad-hoc upsert_one calls intentionally do not).
    try:
        from app.rag.metadata_store import get_metadata_store
        get_metadata_store().record_update({"table_info": loader.last_changed_count})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[table_info] failed to record metadata: %s", exc)


if __name__ == "__main__":
    main()
