"""
BM25 keyword index for the ``table_catalog`` collection.

Public API:
    ``KeywordIndexService``  singleton service (lock / lazy get / warm() / invalidate())
    ``get_keyword_index_service()``  module-level singleton accessor
    ``Bm25Index``            the immutable, query-only index
    ``build_from_collection(collection)``  build from a live ChromaDB collection

CLI:
    python -m app.rag.keyword_search "query"
    python -m app.rag.keyword_search --dump-vocab
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer

from app.core.logger import get_logger
from app.rag.search import COLLECTION_NAME, get_search_config

logger = get_logger(__name__)

# ----------------------------------------------------------------------------
# Cache mechanics. Every *search* knob (weights, depths, k1/b, fields) lives in
# `search.SearchConfig` and arrives as `cfg`.
# ----------------------------------------------------------------------------
#Use pickle for index cache on/off
_CFG_INDEX_CACHE        = os.getenv("KEYWORD_INDEX_CACHE", "1") != "0"
#Version for cache fingerprint to prevent broken deserialization
_CFG_CACHE_FORMAT       = "v8"

_RE_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
# Zero-width camelCase + digit-letter boundaries
_RE_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"      # lower/digit → upper
    r"|(?<=[a-zA-Z])(?=[0-9])"     # letter → digit
    r"|(?<=[0-9])(?=[a-zA-Z])"     # digit → letter
)


def _split_identifier(token: str) -> list[str]:
    """Split a schema.table or plain identifier into the whole form plus its parts"""
    results: list[str] = []

    # strip schema prefix but emit the schema token
    if "." in token:
        parts = token.split(".", 1)
        results.append(parts[0].lower())
        token = parts[1]

    normalized = token.lower()
    if normalized:
        results.append(normalized)  # whole identifier

    # split on non-alnum
    chunks = _RE_NON_ALNUM.split(token)
    for chunk in chunks:
        if not chunk:
            continue
        chunk_lower = chunk.lower()
        if chunk_lower != normalized:
            results.append(chunk_lower)
        # camel/digit boundary split
        sub = _RE_CAMEL_BOUNDARY.sub(" ", chunk).split()
        for s in sub:
            s = s.lower()
            if s != chunk_lower:
                results.append(s)

    return results


def tokenize(text: str) -> list[str]:
    """Split text into normalized tokens for BM25 indexing and querying. Stopwords kept on purpose"""
    out: list[str] = []
    for raw in text.split():
        for token in _split_identifier(raw):
            token = token.lower()
            # 1-char tokens are noise, except digits — years like `2024` count.
            if len(token) <= 1 and not token.isdigit():
                continue
            out.append(token)
    return out


def _aliases_of(meta: dict) -> list[str]:
    """Get table aliases from vector DB metadata as json array"""
    raw = meta.get("table_alias") or "[]"
    try:
        aliases = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    return [str(a) for a in aliases or []]


def _build_keyword_doc(meta: dict, fields: set[str]) -> str:
    """Build the text a table is indexed under, from the metadata fields cfg selects"""
    parts: list[str] = []
    if "name" in fields:
        parts.append(meta.get("table_name") or "")
    if "alias" in fields:
        parts.extend(_aliases_of(meta))
    if "domain" in fields:
        dm = meta.get("domain_mapping") or ""
        if dm:
            parts.append(dm)
    if "description" in fields:
        desc = meta.get("table_description") or ""
        if desc:
            parts.append(desc)
    if "columns" in fields:
        parts.extend(_column_names(meta))
    if "rules" in fields:
        # Business vocabulary ("fronting", join keys) that no other field carries
        rules = meta.get("table_specific_rules") or ""
        if rules:
            parts.append(rules)
    return " ".join(parts)


def _column_names(meta: dict) -> list[str]:
    """Get column names from vector DB metadata as json array"""
    return [str(name) for name in json.loads(meta.get("column_names") or "[]")]


def _persona_ids_of(meta: dict) -> list[int]:
    """Get the persona ids a table row is tagged with, from vector DB metadata as json array"""
    raw = meta.get("persona_ids")
    if not raw:
        return []
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        logger.warning(
            "[keyword_search] Unreadable persona_ids %r — row left untagged.", raw,
        )
        return []
    return [int(p) for p in ids or []]


class _PersonaMasks:
    """Boolean row mask per persona, built once at index build time.

    mask_for(None) means no filtering; an unknown persona gets an all-False mask,
    deliberately not "everything".
    """

    def __init__(self, personas: "list[list[int]]") -> None:
        self._n = len(personas)
        # Keyed by persona, not by row: one dense bool per catalog row, so
        # _masks[5][i] is "can persona 5 see row i". Lets _top_k filter with a
        # single np.where instead of a per-query Python loop over the catalog.
        self._masks: "dict[int, np.ndarray]" = {}
        for row, pids in enumerate(personas):
            for pid in pids:
                mask = self._masks.get(pid)
                if mask is None:
                    mask = self._masks[pid] = np.zeros(self._n, dtype=bool)
                mask[row] = True

    def mask_for(self, persona_id: Optional[int]) -> "Optional[np.ndarray]":
        if persona_id is None:
            return None
        mask = self._masks.get(persona_id)
        return mask if mask is not None else np.zeros(self._n, dtype=bool)


def _top_k(
    scores: "np.ndarray",
    ids: list[str],
    top_n: int,
    mask: "Optional[np.ndarray]" = None,
) -> list[tuple[str, float]]:
    """Apply the persona mask and return the top_n highest-scoring (id, score) pairs"""
    # scores: one BM25 score per catalog row, the query's term weights summed; 0 = no match
    # ids:    table ids in the same row order, so ids[i] names the table scores[i] scored
    # mask:   True where the persona may see the row; None means no filtering
    # All three span the whole catalog, so every index below is a row position
    # into them, not a table id. top_n caps the output, which may be shorter.
    # Zero rather than delete, so indices stay aligned with `ids`
    if mask is not None:
        scores = np.where(mask, scores, 0.0)

    # A document the query never touched is not a result
    nnz = int((scores > 0).sum())
    if nnz == 0:
        return []

    # Bounded by nnz too: argpartition raises past the array length
    k = min(top_n, nnz)
    # argpartition is O(n); the sort then runs over k rows, not the whole catalog
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return [(ids[i], float(scores[i])) for i in top]


# ----------------------------------------------------------------------------
# BM25 Index (5.4)
# ----------------------------------------------------------------------------

class Bm25Index:
    """Immutable Okapi BM25 index (Lucene IDF variant)."""

    def __init__(
        self,
        ids: list[str],
        docs: list[str],
        personas: "list[list[int]]",
        k1: float,
        b: float,
    ) -> None:
        self._ids = ids
        self._personas = _PersonaMasks(personas)
        self._vocab: dict[str, int] = {}
        self._W: Optional[csc_matrix] = None
        # Queries must be analyzed exactly as the documents were
        self._analyzer = tokenize

        if not ids:
            return

        cv = CountVectorizer(analyzer=self._analyzer, dtype=np.float32)
        tf_mat = cv.fit_transform(docs)   # (n_docs, n_terms) CSR

        self._vocab = cv.vocabulary_

        n_docs, n_terms = tf_mat.shape
        doc_len = np.asarray(tf_mat.sum(axis=1)).ravel()
        avgdl = max(float(doc_len.mean()), 1.0)

        # document frequency
        df = np.asarray((tf_mat > 0).sum(axis=0)).ravel()
        # Lucene IDF
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        # Build BM25 weight matrix in COO then store as CSC, vectorised over the non-zeros
        # W[d,t] = idf[t] * tf*(k1+1) / (tf + k1*(1 - b + b*len_d/avgdl))
        cx = tf_mat.tocoo()
        norm = k1 * (1.0 - b + b * (doc_len / avgdl))
        data = (
            idf[cx.col] * cx.data * (k1 + 1.0) / (cx.data + norm[cx.row])
        ).astype(np.float32, copy=False)

        self._W = csc_matrix(
            (data, (cx.row, cx.col)), shape=(n_docs, n_terms), dtype=np.float32
        )

    def search(
        self,
        query: str,
        top_n: int = 10,
        persona_id: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """Return ``(id, score)`` pairs sorted descending."""
        if self._W is None:
            return []

        qtokens = list(dict.fromkeys(self._analyzer(query)))  # distinct, preserve order
        col_indices = [self._vocab[t] for t in qtokens if t in self._vocab]
        if not col_indices:
            return []

        scores = np.asarray(self._W[:, col_indices].sum(axis=1)).ravel()
        return _top_k(scores, self._ids, top_n, self._personas.mask_for(persona_id))

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab


# ----------------------------------------------------------------------------
# Fingerprint / cache helpers
# ----------------------------------------------------------------------------

def _index_fingerprint(collection_data: dict, cfg) -> str:
    """SHA-256 over the catalog rows and the config the cached matrices were built from.

    Code is not covered: edit tokenize() and bump _CFG_CACHE_FORMAT by hand.
    Query-time-only weights are left out, they shape no matrix.
    """
    ids = collection_data.get("ids") or []
    metas = collection_data.get("metadatas") or []
    index_cfg = {
        "cache_format": _CFG_CACHE_FORMAT,
        "fields": sorted(cfg.keyword_fields),
        "k1": cfg.bm25_k1,
        "b": cfg.bm25_b,
    }
    h = hashlib.sha256()
    h.update(json.dumps(index_cfg, sort_keys=True).encode())
    # Sort by id for determinism
    for _id, meta in sorted(zip(ids, metas), key=lambda x: x[0]):
        h.update(_id.encode())
        h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()


def _cache_path() -> Path:
    """Get the pickle cache path, defaulting to the same data dir as chroma_db.py"""
    storage = os.getenv("VECTORDB_STORAGE_PATH") or str(
        Path(__file__).resolve().parents[2] / "data"
    )
    return Path(storage) / "keyword_index" / "table_catalog.pkl"


def _load_from_cache(fingerprint: str) -> Optional[Bm25Index]:
    if not _CFG_INDEX_CACHE:
        return None
    p = _cache_path()
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("fingerprint") == fingerprint:
            logger.info("[keyword_search] Loaded from cache: %s", p)
            return cached["index"]
        logger.info("[keyword_search] Cache fingerprint mismatch — rebuilding.")
    except Exception as exc:
        logger.warning("[keyword_search] Failed to load cache (%s) — rebuilding.", exc)
    return None


def _save_to_cache(index: Bm25Index, fingerprint: str) -> None:
    if not _CFG_INDEX_CACHE:
        return
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"fingerprint": fingerprint, "index": index}, fh)
        tmp.replace(p)
        logger.info("[keyword_search] Index cached at: %s", p)
    except Exception as exc:
        logger.warning("[keyword_search] Could not write cache (%s).", exc)


# ----------------------------------------------------------------------------
# Build from ChromaDB collection
# ----------------------------------------------------------------------------

def build_from_collection(collection, cfg=None) -> Bm25Index:
    """Fetch all rows from a ChromaDB VectorCollection and fit the BM25 index."""
    cfg = cfg or get_search_config()
    logger.info("[keyword_search] Fetching all rows from collection for keyword index build ...")
    data = collection.get(include=["metadatas"])
    ids: list[str] = data.get("ids") or []
    metas: list[dict] = data.get("metadatas") or []
    logger.info("[keyword_search] Fetched %d rows.", len(ids))

    fingerprint = _index_fingerprint(data, cfg)
    cached = _load_from_cache(fingerprint)
    if cached is not None:
        return cached

    docs = [_build_keyword_doc(m, cfg.keyword_fields) for m in metas]
    personas = [_persona_ids_of(m) for m in metas]

    logger.info("[keyword_search] Building BM25 index over %d documents ...", len(ids))
    index = Bm25Index(ids, docs, personas, cfg.bm25_k1, cfg.bm25_b)
    logger.info("[keyword_search] BM25 index built. Vocab size: %d", len(index.vocab))

    _save_to_cache(index, fingerprint)
    return index


# ----------------------------------------------------------------------------
# KeywordIndexService — singleton with lock / lazy / warm / invalidate
# ----------------------------------------------------------------------------

class KeywordIndexService:
    """Thread-safe lazy singleton wrapping the lexical index."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: Optional[Bm25Index] = None

    def get(self) -> Optional[Bm25Index]:
        """Return the current index (may be None if not yet built)."""
        return self._index

    def warm(self, collection=None, cfg=None) -> bool:
        """Build or load the index if not already built, opening the collection if not given"""
        cfg = cfg or get_search_config()
        if not cfg.keyword_weight:
            logger.info("[keyword_search] keyword_weight is zero — skipping warm.")
            return True

        if self._index is not None:
            return True

        with self._lock:
            if self._index is not None:
                return True
            try:
                if collection is None:
                    collection = self._open_collection()
                if collection is None:
                    return False
                idx = build_from_collection(collection, cfg)
                self._index = idx
                logger.info("[keyword_search] warm() complete.")
                return True
            except Exception as exc:
                logger.error("[keyword_search] warm() failed: %s", exc, exc_info=True)
                return False

    def invalidate(self) -> None:
        """Drop the in-memory index (cache file is NOT deleted)."""
        with self._lock:
            self._index = None
        logger.info("[keyword_search] In-memory index invalidated.")

    def _open_collection(self):
        try:
            from app.rag.vector_store import get_collection
            from app.rag.embedding_vertex import get_vectordb_embedding_fn
            return get_collection(
                COLLECTION_NAME,
                get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT"),
            )
        except Exception as exc:
            logger.error("[keyword_search] Could not open collection: %s", exc)
            return None


_svc: Optional[KeywordIndexService] = None
_svc_lock = threading.Lock()


def get_keyword_index_service() -> KeywordIndexService:
    global _svc
    if _svc is None:
        with _svc_lock:
            if _svc is None:
                _svc = KeywordIndexService()
    return _svc


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    import sys
    from app.rag.chroma_db import bootstrap_standalone
    bootstrap_standalone()

    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    if not args:
        print('Usage: python -m app.rag.keyword_search "query text"')
        print('       python -m app.rag.keyword_search --dump-vocab')
        return

    svc = get_keyword_index_service()
    svc.warm()
    idx = svc.get()
    if idx is None:
        print("Index not available.")
        return

    if "--dump-vocab" in args:
        for term in sorted(idx.vocab.keys()):
            print(term)
        return

    query = " ".join(args)
    hits = idx.search(query, top_n=10)
    print(f'\nKeyword search: "{query}"')
    print("=" * 60)
    for rank, (tid, score) in enumerate(hits, 1):
        print(f"  [{rank:2d}]  {tid}  (score={score:.4f})")


if __name__ == "__main__":
    main()

