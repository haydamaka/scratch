"""
Loads example questions (QnA pairs) into the ChromaDB ``question_catalog``
collection for semantic question retrieval / few-shot example selection.

Public API (``QuestionInfoLoader``):
    ``upsert_one(item)``      Upsert a single QnA record. Idempotent.
    ``load(retriever)``       Async. Bulk-upsert everything a retriever returns.
    ``load_from_api()``       Async. Shorthand for the service retriever.
    ``load_from_json(path)``  Blocking. Shorthand for a file retriever.

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
import sys
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

COLLECTION_NAME = "question_catalog"

class QuestionInfoLoader:

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
            if _id not in existing_docs or existing_docs.get(_id) != doc or existing_meta.get(_id) != meta:
                changed += 1
        return changed

    def _get_or_create_collection(self) -> VectorCollection:
        logger.info(
            "[questions] opening/creating vector collection '%s' at '%s' ...",
            COLLECTION_NAME, store_location(),
        )
        embed_fn = get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT")
        collection = get_or_create_collection(COLLECTION_NAME, embed_fn)
        logger.info("[questions] vector collection '%s' ready.", COLLECTION_NAME)
        return collection

    @staticmethod
    def _question_id(item: dict) -> str:
        """Deterministic id derived from the (stripped) question text."""
        question = (item.get("question") or "").strip()
        return "q_" + hashlib.sha1(question.encode("utf-8")).hexdigest()

    def _prepare_records(self, records: list[dict]) -> "tuple[list[dict], int]":
        """
        Drop records with an empty question and de-duplicate by question id
        (last occurrence wins).

        Returns ``(clean_records, dropped_count)``.
        """
        by_id: dict[str, dict] = {}
        dropped = 0
        for it in records:
            if not (it.get("question") or "").strip():
                dropped += 1
                continue
            by_id[self._question_id(it)] = it
        return list(by_id.values()), dropped

    def _partition_by_existence(
        self,
        collection: VectorCollection,
        items: list[dict],
    ) -> "tuple[list[dict], list[str]]":
        """
        Split ``items`` into ``(to_upsert, skipped_ids)`` according to
        ``UPDATE_EXISTING``.
        """
        items = list(items)
        if UPDATE_EXISTING or not items:
            return items, []

        fetched = collection.get(
            ids=[self._question_id(it) for it in items],
            include=["metadatas"],
        )
        existing_ids = set(fetched.get("ids") or [])

        to_upsert: list[dict] = []
        skipped_ids: list[str] = []
        for it in items:
            qid = self._question_id(it)
            if qid in existing_ids:
                skipped_ids.append(qid)
            else:
                to_upsert.append(it)
        return to_upsert, skipped_ids

    def _build_document(self, item: dict) -> str:
        """The embedded document is the question text only."""
        question = (item.get("question") or "").strip()
        return question[:DOCUMENT_MAX_CHARS]

    def _build_metadata(self, item: dict) -> dict:
        """
        Produce a flat ChromaDB-safe metadata dict from one QnA record.
        Lists are JSON-encoded; convenience scalars are derived for readability.
        """
        expected = item.get("expectedTable") or []
        table_names = [
            str(t.get("tableName"))
            for t in expected
            if isinstance(t, dict) and t.get("tableName")
        ]
        domains = [
            str(t.get("domain"))
            for t in expected
            if isinstance(t, dict) and t.get("domain")
        ]

        ground_truth_sql = item.get("groundTruthSql") or ""

        persona_ids = list({
            pid
            for t in expected
            if isinstance(t, dict) and t.get("persona_id") is not None
            for pid in (t["persona_id"] if isinstance(t["persona_id"], list) else [t["persona_id"]])
        })

        metadata = {
            "question":         item.get("question") or "",
            "ground_truth_sql": ground_truth_sql,
            "expected_table":   json.dumps(expected),
            "table_names":      ", ".join(table_names),
            "domains":          ", ".join(domains),
            "persona_ids":      json.dumps(persona_ids),
        }

        # Backend-friendly membership flags (works with both Chroma and Milvus filters).
        for pid in persona_ids:
            metadata[f"persona_id_{pid}"] = 1

        return metadata


    def upsert_one(self, item: dict) -> None:
        question_id = self._question_id(item)

        if not (item.get("question") or "").strip():
            logger.warning("[questions] upsert_one: empty question — skipping.")
            return

        collection = self._get_or_create_collection()

        if not UPDATE_EXISTING:
            _, skipped_ids = self._partition_by_existence(collection, [item])
            if skipped_ids:
                logger.info(
                    "[questions] upsert_one: '%s' already exists — "
                    "skipping (update_existing=False).",
                    question_id,
                )
                return

        logger.info("[questions] upsert_one: building document for '%s' ...", question_id)

        document = self._build_document(item)
        metadata = self._build_metadata(item)

        logger.debug(
            "[questions] upsert_one '%s': document=%d chars, tables=[%s]",
            question_id, len(document), metadata["table_names"] or "(none)",
        )

        count_before = collection.count()

        collection.upsert(
            ids       = [question_id],
            documents = [document],
            metadatas = [metadata],
        )

        count_after = collection.count()
        action = "inserted" if count_after > count_before else "updated"
        logger.info(
            "[questions] upsert_one: '%s' %s → collection '%s' "
            "(collection size: %d → %d).",
            question_id, action, COLLECTION_NAME, count_before, count_after,
        )

    def load_from_json(
        self,
        json_path: "str | Path | None" = None,
        *,
        max_records: Optional[int] = None,
        show_progress: bool = True,
    ) -> VectorCollection:
        """Load from a file. Synchronous for callers that are not in an event
        loop; inside one, await ``load(LocalQuestionsRetriever(path))`` instead."""
        from app.rag.local_questions_retriever import LocalQuestionsRetriever

        return asyncio.run(self.load(
            LocalQuestionsRetriever(json_path),
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
        logger.info("[questions] load: reading from %s ...", retriever.source)
        qna_records = await retriever.fetch()

        qna_records, dropped = self._prepare_records(qna_records)
        total_in_source = len(qna_records)
        logger.info(
            "[questions] load: received %d usable records (%d dropped as empty/dupe).",
            total_in_source, dropped
        )

        if max_records is not None:
            qna_records = qna_records[:max_records]
            logger.info(
                "[questions] max_records=%d → processing %d of %d records.",
                max_records, len(qna_records), total_in_source,
            )

        collection   = self._get_or_create_collection()
        count_before = collection.count()
        logger.info(
            "[questions] Collection '%s' current size: %d documents.",
            COLLECTION_NAME, count_before,
        )

        total         = len(qna_records)
        upserted      = 0
        skipped_total = 0
        changed_total = 0

        batch_size = upsert_batch_size()
        for batch_start in range(0, total, batch_size):
            batch = qna_records[batch_start : batch_start + batch_size]

            if not UPDATE_EXISTING:
                batch, skipped_ids = self._partition_by_existence(collection, batch)
                if skipped_ids:
                    skipped_total += len(skipped_ids)
                    logger.info(
                        "[questions] batch @%d: skipping %d existing record(s) "
                        "(update_existing=False).",
                        batch_start + 1, len(skipped_ids),
                    )
                if not batch:
                    continue

            ids, documents, metadatas = [], [], []
            for item in batch:
                doc  = self._build_document(item)
                meta = self._build_metadata(item)
                ids.append(self._question_id(item))
                documents.append(doc)
                metadatas.append(meta)
                logger.debug(
                    "[questions]   '%s': doc=%d chars, tables=[%s]",
                    self._question_id(item), len(doc), meta["table_names"] or "(none)",
                )

            n_changed = self._count_changed(collection, ids, documents, metadatas)

            try:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                upserted += len(batch)
                # Credited only once the write lands, so a failed batch does not
                # report records it never stored.
                changed_total += n_changed
            # Fail-safe, not fail-fast: one bad batch is isolated to its own
            # records and the rest of the catalog still lands. A timeout takes
            # this path too — a slow endpoint should cost the records it touched,
            # not the whole load.
            except Exception as batch_exc:
                logger.error(
                    "[questions] batch [%d-%d] FAILED (%s). "
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
                        logger.error("[questions]   SKIP '%s': %s", doc_id, single_exc)
                logger.info(
                    "[questions] batch recovery: %d ok, %d skipped.", n_ok, n_fail
                )

            if show_progress:
                logger.info(
                    "[questions] batch [%d-%d]: %d/%d upserted.",
                    batch_start + 1, batch_start + len(batch), upserted, total,
                )

        count_after = collection.count()
        logger.info(
            "[questions] load: done. upserted=%d, skipped=%d, "
            "collection size: %d → %d (%+d). Store: %s",
            upserted, skipped_total, count_before, count_after,
            count_after - count_before, store_location(),
        )
        self.last_changed_count = changed_total
        logger.info("[questions] load: %d record(s) actually changed.", changed_total)
        return collection

    async def load_from_api(
        self,
        *,
        max_records: Optional[int] = None,
        show_progress: bool = True,
    ) -> VectorCollection:
        """Load from the training-intelligence service."""
        from app.rag.questions_retriever import QuestionsRetriever

        return await self.load(
            QuestionsRetriever(), max_records=max_records, show_progress=show_progress,
        )



def main() -> None:
    """
    Example::
        python -m app.rag.questions_loader
    """
    bootstrap_standalone()

    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logger.info("[questions] CLI load mode: loading from API.")
    loader = QuestionInfoLoader()
    asyncio.run(loader.load_from_api())

    # A full CLI load is a real bulk load → record the DB version/metadata
    # (ad-hoc upsert_one calls intentionally do not).
    try:
        from app.rag.metadata_store import get_metadata_store
        get_metadata_store().record_update({"questions": loader.last_changed_count})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[questions] failed to record metadata: %s", exc)


if __name__ == "__main__":
    main()
