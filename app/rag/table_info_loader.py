"""
Loads table metadata from the persona API into the ChromaDB ``table_catalog``
collection for semantic table pre-filtering.

Public API (``TableInfoLoader``):
    ``upsert_one(item)``      Upsert a single table record. Idempotent.
    ``load_from_api()``       Async. Bulk-upsert all records from the persona API.
    ``load_from_json(path)``  Load from a JSON file (offline testing).

CLI:  python -m app.rag.table_info_loader
"""

from __future__ import annotations

import os

# Use pysqlite3 (sqlite >= 3.35) before any chromadb import.
if os.name == "posix":
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import warnings

# Silence noisy third-party startup warnings (must precede third-party imports).
from authlib.deprecate import AuthlibDeprecationWarning as _AuthlibDeprecationWarning
warnings.filterwarnings("ignore", category=_AuthlibDeprecationWarning)
warnings.filterwarnings("ignore", module="requests")

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

from agent.services.tableschema_persona import TableSchemaPersonaDetails
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
    UPSERT_BATCH_SIZE,
)

logger = get_logger(__name__)

COLLECTION_NAME = "table_catalog"

_DEFAULT_JSON_PATH = (
    Path(__file__).resolve().parent / "sample-data" / "table_data.json"
)


class TableInfoLoader:
    """Builds and maintains the ChromaDB ``table_catalog`` collection."""

    # Number of records actually changed (new or differing) by the last load.
    last_changed_count: int = 0

    @staticmethod
    def _count_changed(
        collection: VectorCollection,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> int:
        """Count how many of the given records are new or differ from what's
        already stored (so callers can tell whether a load really changed data)."""
        if not ids:
            return 0
        try:
            fetched = collection.get(ids=ids, include=["documents", "metadatas"])
        except Exception:  # pragma: no cover - defensive; treat as all-changed
            return len(ids)

        existing_docs = {
            _id: doc
            for _id, doc in zip(fetched.get("ids") or [], fetched.get("documents") or [])
        }
        existing_meta = {
            _id: meta
            for _id, meta in zip(fetched.get("ids") or [], fetched.get("metadatas") or [])
        }

        changed = 0
        for _id, doc, meta in zip(ids, documents, metadatas):
            if _id not in existing_docs:
                changed += 1
            elif existing_docs.get(_id) != doc or existing_meta.get(_id) != meta:
                changed += 1
        return changed

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
        values = []
        if raw_persona_ids is None:
            return values

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

    @staticmethod
    def _as_scalar_str(value) -> str:
        """Return a stable string for metadata fields that may arrive as list/dict."""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, list):
            # Typical upstream shape: ["Loans Instruction"]
            return ", ".join(str(v) for v in value if v is not None)
        if isinstance(value, dict):
            # Preserve structure while staying Chroma-safe.
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        return str(value)

    @classmethod
    def _as_scalar_int(cls, value, default: int = 0) -> int:
        """Best-effort int conversion for metadata fields like version."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            try:
                return int(float(text))
            except ValueError:
                return default
        if isinstance(value, list) and value:
            return cls._as_scalar_int(value[0], default=default)
        return default

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

    @staticmethod
    def _column_names(item: dict) -> list[str]:
        """Column names of one table record; entries may be strings or dicts."""
        names: list[str] = []
        for c in item.get("columns") or []:
            if isinstance(c, str):
                if c:
                    names.append(c)
            elif isinstance(c, dict):
                name = c.get("columnName")
                if name:
                    names.append(str(name))
        return names

    def _build_document(self, item: dict) -> str:
        """
        Build a plain-text string for embedding from one table record.
          - Table name
          - Aliases (if present)
          - Description
          - Not embedded: column names
        """
        parts: list[str] = []

        parts.append(f"TABLE: {item['tableName']}")

        aliases = item.get("tableAlias") or []
        if aliases:
            parts.append(f"ALIASES: {', '.join(str(a) for a in aliases)}")

        parts.append(f"DESCRIPTION: {item.get('tableDescription') or ''}")

        return "\n".join(parts)[:DOCUMENT_MAX_CHARS]

    def _build_metadata(self, item: dict) -> dict:
        """
        Produce a flat ChromaDB-safe metadata dict from one table record.
        """
        persona_ids = self._persona_ids_of(item)
        metadata = {
            "table_name":         item["tableName"],
            "table_description": self._as_scalar_str(item.get("tableDescription")),
            "version":           self._as_scalar_int(item.get("version"), default=0),
            "table_specific_rules": self._as_scalar_str(item.get("tableSpecificRules")),
            "domain_mapping":       self._as_scalar_str(item.get("domainMapping")),
            "persona_name":         self._as_scalar_str(item.get("personaName")),
            "persona_id": self._primary_persona_id(persona_ids),
            "persona_ids": json.dumps(persona_ids),
            "table_alias": json.dumps(item.get("tableAlias") or []),
            # Carries the column retriever of the keyword index. Untruncated on
            # purpose: the DOCUMENT_MAX_CHARS budget exists for the embedder's
            # token limit, and BM25 has no equivalent — truncating here would
            # drop exactly the rare column names IDF weights highest.
            "columns": ", ".join(self._column_names(item)),
        }
        for pid in persona_ids:
            metadata[f"persona_id_{pid}"] = 1
        return metadata


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
        json_path: "str | Path" = _DEFAULT_JSON_PATH,
        *,
        max_records: Optional[int] = 10,
        show_progress: bool = True,
    ) -> VectorCollection:
        json_path = Path(json_path).resolve()

        logger.info("[table_info] reading JSON from %s", json_path)
        logger.info("[table_info] file size: %.1f KB", json_path.stat().st_size / 1_024)

        with open(json_path, "r", encoding="utf-8") as fh:
            all_records: list[dict] = json.load(fh)

        total_in_file = len(all_records)
        logger.info("[table_info] JSON contains %d records total.", total_in_file)

        if max_records is not None:
            table_records = all_records[:max_records]
            logger.info(
                "[table_info] max_records=%d → processing %d of %d records.",
                max_records, len(table_records), total_in_file,
            )
        else:
            table_records = all_records
            logger.info(
                "[table_info] max_records=None → processing all %d records.",
                total_in_file,
            )

        logger.info(
            "[table_info] opening vector collection '%s' at '%s'",
            COLLECTION_NAME, store_location(),
        )

        collection   = self._get_or_create_collection()
        count_before = collection.count()
        logger.info(
            "[table_info] Collection '%s' current size: %d documents.",
            COLLECTION_NAME, count_before,
        )

        logger.info(
            "[table_info] upserting %d records (batch_size=%d, update_existing=%s) ...",
            len(table_records), UPSERT_BATCH_SIZE, UPDATE_EXISTING,
        )

        total         = len(table_records)
        upserted      = 0
        skipped_total = 0

        for batch_start in range(0, total, UPSERT_BATCH_SIZE):
            batch = table_records[batch_start : batch_start + UPSERT_BATCH_SIZE]

            # —— skip records that already exist when updates are disabled ——
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
                ids.append(item["tableName"])
                documents.append(self._build_document(item))
                metadatas.append(self._build_metadata(item))

            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            upserted += len(batch)

            if show_progress:
                logger.info(
                    "[table_info] batch [%d-%d]: %d/%d upserted.",
                    batch_start + 1, batch_start + len(batch), upserted, total,
                )

        count_after = collection.count()
        new_docs    = count_after - count_before
        logger.info(
            "[table_info] done. upserted=%d, skipped=%d, collection size: %d → %d "
            "(%+d new). Store: %s",
            upserted, skipped_total, count_before, count_after, new_docs, store_location(),
        )

        return collection

    async def load_from_api(
        self,
        *,
        max_records: Optional[int] = None,
        show_progress: bool = True,
    ) -> VectorCollection:
        logger.info("[table_info] load_from_api: fetching data via get_table_persona_schema() ...")
        table_records = await TableSchemaPersonaDetails().get_table_persona_schema()

        total_in_source = len(table_records)
        logger.info("[table_info] load_from_api: received %d records from API.", total_in_source)

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

        total         = len(table_records)
        upserted      = 0
        skipped_total = 0
        changed_total = 0

        for batch_start in range(0, total, UPSERT_BATCH_SIZE):
            batch = table_records[batch_start : batch_start + UPSERT_BATCH_SIZE]

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

            changed_total += self._count_changed(collection, ids, documents, metadatas)

            try:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                upserted += len(batch)
            except TimeoutError:
                logger.error(
                    "[table_info] batch [%d-%d] TIMED OUT while embedding — endpoint "
                    "unreachable/slow; aborting load fast (loader will retry per policy, "
                    "then /readyz stays 503). Not retrying one-by-one.",
                    batch_start + 1, batch_start + len(batch),
                )
                raise
            except Exception as batch_exc:
                logger.error(
                    "[table_info] batch [%d-%d] FAILED (%s). "
                    "Retrying %d docs one-by-one ...",
                    batch_start + 1, batch_start + len(batch),
                    batch_exc, len(batch),
                )
                n_ok = n_fail = 0
                for doc_id, doc, meta in zip(ids, documents, metadatas):
                    try:
                        collection.upsert(ids=[doc_id], documents=[doc], metadatas=[meta])
                        n_ok    += 1
                        upserted += 1
                    except Exception as single_exc:
                        n_fail += 1
                        logger.error("[table_info]   SKIP '%s': %s", doc_id, single_exc)
                logger.info(
                    "[table_info] batch recovery: %d ok, %d skipped.", n_ok, n_fail
                )

            if show_progress:
                logger.info(
                    "[table_info] batch [%d-%d]: %d/%d upserted.",
                    batch_start + 1, batch_start + len(batch), upserted, total,
                )

        count_after = collection.count()
        logger.info(
            "[table_info] load_from_api: done. upserted=%d, skipped=%d, "
            "collection size: %d → %d (%+d). Store: %s",
            upserted, skipped_total, count_before, count_after,
            count_after - count_before, store_location(),
        )
        self.last_changed_count = changed_total
        logger.info("[table_info] load_from_api: %d record(s) actually changed.", changed_total)
        try:
            from app.rag.keyword_index import get_keyword_index_service
            get_keyword_index_service().invalidate()
        except Exception as _kw_exc:
            logger.debug("[table_info] keyword index invalidation skipped: %s", _kw_exc)
        return collection



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
