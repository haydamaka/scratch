"""
Background loader + readiness gate for the vector-store collections
CLI:  python -m app.rag.vectordb_loader
"""

from __future__ import annotations

import os
import sys

# ----------------------------------------------
# Config parameters
# ----------------------------------------------
_STOP_JOIN_TIMEOUT_SECONDS = float(os.getenv("VECTORDB_LOAD_STOP_TIMEOUT_SECONDS", "10"))
# How long a search waits for the startup load before giving up on it. Bounded so
# a slow or wedged load degrades the first queries instead of hanging them.
_READY_WAIT_TIMEOUT_SECONDS = float(os.getenv("VECTORDB_READY_TIMEOUT_SECONDS", "120"))


def _reload_enabled() -> bool:
    return os.getenv("VECTORDB_RELOAD", "") == "true"


if os.name == "posix":
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import asyncio
import threading
from enum import Enum
from typing import Optional

from app.core.logger import get_logger
from app.rag.table_info_loader import TableInfoLoader
from app.rag.questions_loader import QuestionInfoLoader

logger = get_logger(__name__)

class LoadState(str, Enum):
    LOADING = "loading"
    READY   = "ready"
    FAILED  = "failed"


class VectorStoreLoaderService:

    def __init__(self) -> None:
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state      = LoadState.LOADING
        # Set once _run() finishes, however it finished. Readers block on this
        # rather than on the thread, so a failed load releases them too.
        self._ready      = threading.Event()

    @property
    def is_ready(self) -> bool:
        return self._state is LoadState.READY

    @property
    def state(self) -> LoadState:
        return self._state

    def start(self, *, force: bool = False) -> None:
        if self._running:
            logger.warning("[vectordb] loader already running — ignoring start().")
            return


        self._running = True
        self._stop_event.clear()
        self._ready.clear()
        self._state = LoadState.LOADING

        self._thread = threading.Thread(
            target=self._run, name="vectordb-loader", daemon=True
        )
        self._thread.start()
        logger.info("[vectordb] loader started (background thread, own event loop).")

    def stop(self) -> None:
        if not self._running:
            return
        logger.info("[vectordb] stopping loader ...")
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_SECONDS)
        logger.info("[vectordb] loader stopped.")

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            logger.info("[vectordb] load starting ...")
            load_ok = True
            if _reload_enabled():
                try:
                    load_ok = asyncio.run(self._load_all())
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("[vectordb] load crashed: %s", exc)
                    load_ok = False
            else:
                logger.info(
                    "[vectordb] VECTORDB_RELOAD not set — skipping catalog load; "
                    "serving existing persisted store."
                )

            if load_ok:
                # After the load, never beside it: the keyword index is fitted from
                # the collection, so building it concurrently would index a
                # half-written catalog.
                try:
                    from app.rag.keyword_index import get_keyword_index_service
                    get_keyword_index_service().warm()
                except Exception as exc:
                    logger.error("[vectordb] keyword index warm() failed: %s", exc)

                self._state = LoadState.READY
                logger.info("[vectordb] ✅ startup complete — state=READY.")
            else:
                self._state = LoadState.FAILED
                logger.error("[vectordb] ❌ load failed — state=FAILED.")
        finally:
            # In `finally` so a crash releases waiters instead of hanging them.
            self._ready.set()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the startup load has finished. True when it ended READY."""
        if not self._ready.wait(timeout):
            logger.warning(
                "[vectordb] still loading after %.0fs — proceeding without it.", timeout,
            )
            return False
        return self._state is LoadState.READY

    async def _load_all(self) -> bool:
        table_loader    = TableInfoLoader()
        question_loader = QuestionInfoLoader()

        results = await asyncio.gather(
            table_loader.load_from_api(),
            question_loader.load_from_api(),
            return_exceptions=True,
        )

        ok = True
        updated_rows: dict[str, int] = {}
        loaders = (
            ("table_info", table_loader),
            ("questions", question_loader),
        )
        for (name, loader), result in zip(loaders, results):
            if isinstance(result, Exception):
                ok = False
                logger.error("[vectordb] load failed for %s: %s", name, result)
            else:
                logger.info("[vectordb] load complete for %s.", name)
                updated_rows[name] = int(getattr(loader, "last_changed_count", 0))

        # Only record metadata when the load actually happened here and changed data.
        if ok:
            try:
                from app.rag.metadata_store import get_metadata_store
                get_metadata_store().record_update(updated_rows)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("[vectordb] failed to record metadata: %s", exc)

        return ok


_service: Optional[VectorStoreLoaderService] = None

def get_vector_store_loader() -> VectorStoreLoaderService:
    global _service
    if _service is None:
        _service = VectorStoreLoaderService()
    return _service


def init_vector_store_loader(*, force: bool = False) -> VectorStoreLoaderService:
    service = get_vector_store_loader()
    service.start(force=force)
    return service


def wait_until_ready(timeout: Optional[float] = _READY_WAIT_TIMEOUT_SECONDS) -> bool:
    """Block callers until the startup load has finished.

    A no-op when no loader was started (CLI, tests, any process that skipped
    ``init_vector_store_loader``) — those paths build what they need lazily and
    must not block on a service that will never run.
    """
    service = _service
    if service is None:
        return True
    return service.wait_ready(timeout)


def stop_vector_store_loader() -> None:
    global _service
    if _service is not None:
        _service.stop()
        _service = None


def main() -> None:
    """Standalone entry point"""
    from app.rag.chroma_db import bootstrap_standalone

    bootstrap_standalone()

    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logger.info("[vectordb] CLI load mode: loading both catalogs from API.")
    service = init_vector_store_loader(force=True)
    service.join()
    logger.info("[vectordb] CLI load finished: state=%s", service.state.value)


if __name__ == "__main__":
    main()
