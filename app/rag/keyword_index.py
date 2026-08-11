"""
BM25 keyword index for the ``table_catalog`` collection.

Public API:
    ``KeywordIndexService``  singleton service (lock / lazy get / warm() / invalidate())
    ``get_keyword_index_service()``  module-level singleton accessor
    ``Bm25Index``            the immutable, query-only index
    ``build_from_collection(collection)``  build from a live ChromaDB collection

CLI:
    python -m app.rag.keyword_index "query"
    python -m app.rag.keyword_index --dump-vocab
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
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

from app.core.logger import get_logger
from app.rag.hybrid_search import COLLECTION_NAME, fuse_rrf, get_search_config

logger = get_logger(__name__)

# ----------------------------------------------------------------------------
# Cache mechanics. Every *search* knob (weights, depths, k1/b, ngram range,
# fields) lives in `hybrid_search.SearchConfig` and arrives as `cfg`.
# ----------------------------------------------------------------------------
_CFG_INDEX_CACHE        = os.getenv("KEYWORD_INDEX_CACHE", "1") != "0"
# Bump either whenever a change to the pickled classes makes an existing cache
# unreadable: an old pickle would otherwise unpickle into the changed class and
# fail at search time.
_CFG_CACHE_FORMAT       = "v4"
_CFG_TABLE_NAME_BOOST_VERSION = "v2"

# ----------------------------------------------------------------------------
# Analyzer (5.2)
# ----------------------------------------------------------------------------
_RE_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")
# Zero-width camelCase + digit-letter boundaries
_RE_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"      # lower/digit → upper
    r"|(?<=[a-zA-Z])(?=[0-9])"     # letter → digit
    r"|(?<=[0-9])(?=[a-zA-Z])"     # digit → letter
)

_DOMAIN_STOPWORDS = {"table", "tbl", "vw", "view", "dim", "fact", "stg", "tmp"}
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | _DOMAIN_STOPWORDS

# Abbreviation → full word, applied to documents and queries alike
_ABBREVIATIONS: dict[str, str] = {
    "acct": "account",
    "amt": "amount",
    "bal": "balance",
    "txn": "transaction",
    "trx": "transaction",
    "cust": "customer",
    "ccy": "currency",
    "curr": "currency",
    "cd": "code",
    "dt": "date",
    "ts": "timestamp",
    "ind": "indicator",
    "flg": "flag",
    "desc": "description",
    "ref": "reference",
    "agmt": "agreement",
    "cpty": "counterparty",
    "ctpy": "counterparty",
    "lgl": "legal",
    "ent": "entity",
    "hier": "hierarchy",
    "mstr": "master",
    "hist": "history",
    "ec": "security",
    "pos": "position",
    "mkt": "market",
    "val": "valuation",
    "exp": "exposure",
    "lmt": "limit",
    "gl": "ledger",
    "org": "organization",
    "addr": "address",
    "num": "number",
    "id": "identifier",
    "qty": "quantity",
    "pct": "percent",
    "yr": "year",
    "mo": "month",
    "wk": "week",
    "seq": "sequence",
    "src": "source",
    "tgt": "target",
    "cfg": "config",
    "typ": "type",
    "cls": "class",
    "cat": "category",
    "grp": "group",
    "lvl": "level",
    "prc": "price",
    "int": "interest",
    "ctry": "country",
    "cntry": "country",
    "dept": "department",
    "mgr": "manager",
    "emp": "employee",
    "acl": "access",
    "auth": "authorization",
    "msg": "message",
    "evt": "event",
    "txr": "transfer",
    "svc": "service",
    "sys": "system",
}


def _split_identifier(token: str) -> list[str]:
    """Split a schema.table or plain identifier into sub-tokens, emitting both
    whole normalized form and its camel/underscore parts."""
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


def _split_and_normalize(text: str, *, drop_stopwords: bool) -> list[str]:
    """Shared body of the two tokenizers; they differ only in stopword handling."""
    out: list[str] = []
    for raw in text.split():
        for token in _split_identifier(raw):
            token = _ABBREVIATIONS.get(token.lower(), token.lower())
            if drop_stopwords and token in _STOPWORDS:
                continue
            # 1-char tokens are noise, except digits — years like `2024` count.
            if len(token) <= 1 and not token.isdigit():
                continue
            out.append(token)
    return out


def tokenize(text: str) -> list[str]:
    """Analyze text into normalized tokens for BM25 indexing / querying.

    Must stay a plain one-argument module-level function: it is handed to
    ``CountVectorizer(analyzer=...)`` and therefore has to be picklable for the
    index cache.
    """
    return _split_and_normalize(text, drop_stopwords=True)


def tokenize_keep_stopwords(text: str) -> list[str]:
    """Tokenize without stopword removal, for the name/alias index.

    Also a module-level function because it is handed to
    ``CountVectorizer(analyzer=...)`` and has to pickle with the index cache.
    """
    return _split_and_normalize(text, drop_stopwords=False)


def _aliases_of(meta: dict) -> list[str]:
    """Table aliases, which metadata stores as a JSON string."""
    raw = meta.get("table_alias") or "[]"
    try:
        aliases = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    return [str(a) for a in aliases or []]


def _build_keyword_doc(meta: dict, fields: set[str]) -> str:
    """Assemble the keyword document string from table metadata."""
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
    if "rules" in fields:
        # `table_specific_rules` carries business vocabulary that appears in no
        # other indexed field and is not embedded either (`_build_document`
        # builds TABLE/ALIASES/DESCRIPTION/COLUMNS) — "fronting", "intercompany"
        # and the join keys live only here.
        rules = meta.get("table_specific_rules") or ""
        if rules:
            parts.append(rules)
    return " ".join(parts)


def _build_column_doc(meta: dict) -> str:
    """Build the lexical document for the column retriever: column names only."""
    return " ".join(part.strip() for part in str(meta.get("columns") or "").split(","))


def _build_name_alias_doc(meta: dict) -> str:
    """Build focused lexical text from table name + aliases only."""
    return " ".join([meta.get("table_name") or "", *_aliases_of(meta)])


def _fuse_or_keep(
    primary: list[tuple[str, float]],
    secondary: list[tuple[str, float]],
    weight: float,
    rrf_k: int,
) -> list[tuple[str, float]]:
    """RRF-fuse two ranked retrievers, degrading to whichever one has hits.

    A zero weight or an empty ``secondary`` leaves ``primary`` untouched — and
    untouched matters: fusing against an empty list would silently replace the
    retriever's own scores with RRF pseudo-scores while keeping the same order.
    """
    if weight <= 0.0 or not secondary:
        return primary
    if not primary:
        return secondary
    return fuse_rrf(
        [h[0] for h in primary],
        [h[0] for h in secondary],
        k=rrf_k,
        w_a=1.0,
        w_b=weight,
    )


def _persona_ids_of(meta: dict) -> list[int]:
    """Every persona a stored row is tagged with.

    Called once per row by ``build_from_collection``; the result is the ``personas``
    list each index turns into :class:`_PersonaMasks`.

    The loaders write both a ``persona_ids`` JSON list and a scalar ``persona_id``
    holding only the first of them. The list is the truth. Falling back to the scalar
    keeps a row findable under one persona rather than none, but hides it from its
    others — hence the warning on that path.
    """
    meta = meta or {}
    raw = meta.get("persona_ids")
    if raw:
        try:
            ids = json.loads(raw) if isinstance(raw, str) else raw
            return [int(p) for p in ids]
        except Exception:
            logger.warning(
                "[keyword_index] Unreadable persona_ids %r — falling back to the "
                "primary id, which hides this table from its other personas.", raw,
            )
    pid = meta.get("persona_id")
    return [int(pid)] if pid is not None else []


class _PersonaMasks:
    """Row masks per persona, precomputed once at build time.

    Held by :class:`Bm25Index` and :class:`NgramIndex`; their ``search`` passes
    ``mask_for(persona_id)`` to :func:`_top_k`, which drops the rows that persona
    cannot see before taking the top hits.

    A dict of boolean arrays rather than a per-query scan over a list of id lists:
    the filter runs on every search, over the whole catalog, and the tags are fixed
    the moment the index is built.

    ``mask_for(None)`` means no filtering. An unknown persona gets an all-False mask —
    no results, deliberately not "everything", which would quietly widen a filtered
    search into an unfiltered one.
    """

    def __init__(self, personas: "list[list[int]]") -> None:
        self._n = len(personas)
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
    """Zero out the rows ``mask`` excludes, then return the ``top_n`` highest.

    The last step of every sparse retriever: turns a score vector over the whole
    catalog into the handful of ``(table_id, score)`` pairs the caller wants.

    Args:
        scores: One score per catalog row, positionally aligned with ``ids``.
            Dense and mostly zero — a query touches few documents.
        ids: Table ids in row order. ``ids[i]`` names the table ``scores[i]`` scored.
        top_n: Maximum number of pairs to return. Fewer come back when fewer
            rows scored above zero.
        mask: Optional boolean row filter, from ``_PersonaMasks.mask_for``. True
            keeps a row. ``None`` means no filtering.

    Returns:
        ``(table_id, score)`` sorted by score descending, at most ``top_n`` long.
    """
    # Zero rather than delete: indices stay aligned with `ids`, so no bookkeeping.
    if mask is not None:
        scores = np.where(mask, scores, 0.0)

    # Zero scores never rank — a document the query did not touch is not a result.
    nnz = int((scores > 0).sum())
    if nnz == 0:
        return []

    # Bounded by nnz as well as top_n: argpartition raises past the array length,
    # and asking for more than nnz would pad the result with zero-scored tables.
    k = min(top_n, nnz)
    # argpartition is O(n) and leaves the top k unordered; the sort then runs over
    # k rows, not the whole catalog.
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
        analyzer=None,
    ) -> None:
        self._ids = ids
        self._personas = _PersonaMasks(personas)
        self._vocab: dict[str, int] = {}
        self._W: Optional[csc_matrix] = None
        # Queries must be analyzed exactly as the documents were: the name index
        # keeps stopwords, so tokenizing its queries with the default analyzer
        # would drop `amount` from the query and never match what it indexed.
        self._analyzer = analyzer or tokenize

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

        # Build BM25 weight matrix in COO then store as CSC.
        # W[d,t] = idf[t] * tf*(k1+1) / (tf + k1*(1 - b + b*len_d/avgdl)).
        # Vectorised over the non-zeros: a Python loop here is O(nnz) interpreted
        # work on every index build, for no gain in clarity.
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
        if self._W is None or not self._ids:
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
# Char-n-gram index (TF-IDF, third retriever)
# ----------------------------------------------------------------------------

class NgramIndex:
    """Char-n-gram TF-IDF index for bridging unmapped abbreviations."""

    def __init__(
        self,
        ids: list[str],
        docs: list[str],
        personas: "list[list[int]]",
        ngram_range: "tuple[int, int]",
    ) -> None:
        self._ids = ids
        self._personas = _PersonaMasks(personas)
        self._vec: Optional[TfidfVectorizer] = None
        self._mat = None  # CSC

        if not ids:
            return

        self._vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(ngram_range[0], ngram_range[1]),
        )
        mat = self._vec.fit_transform(docs)
        self._mat = mat.T.tocsc()  # (n_terms, n_docs) → query slice from rows

    def search(
        self,
        query: str,
        top_n: int = 10,
        persona_id: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        if self._vec is None or self._mat is None or not self._ids:
            return []

        # qvec: (1, n_terms); self._mat: (n_terms, n_docs) -> product: (1, n_docs)
        # Keep this as a plain ndarray to avoid sparse truth-value edge cases.
        qvec = self._vec.transform([query])
        prod = qvec * self._mat
        scores = prod.toarray().ravel().astype(np.float32, copy=False)
        return _top_k(scores, self._ids, top_n, self._personas.mask_for(persona_id))


# ----------------------------------------------------------------------------
# Combined keyword index (BM25 + optional char-ngram)
# ----------------------------------------------------------------------------

class KeywordIndex:
    """Holds a BM25 index plus optional char-n-gram and column retrievers."""

    def __init__(
        self,
        ids: list[str],
        docs: list[str],
        name_alias_docs: list[str],
        column_docs: list[str],
        personas: "list[list[int]]",
        cfg,
    ) -> None:
        self.ids = ids
        # No masks here: persona filtering happens inside each sub-index, and the
        # name-boost scan that once needed them at this level is now a Bm25Index.
        self.bm25 = Bm25Index(ids, docs, personas, cfg.bm25_k1, cfg.bm25_b)
        self.ngram = (
            NgramIndex(ids, docs, personas, cfg.ngram_range)
            if cfg.ngram_weight > 0.0 else None
        )
        # Stopword-keeping analyzer on purpose: `fact`, `dim` and (via sklearn's
        # English list) `amount`, `interest`, `name` are dropped by `tokenize`,
        # and those are exactly the words table names are made of. IDF weights
        # them by how common they actually are in this catalog.
        self.name_alias_bm25 = Bm25Index(
            ids, name_alias_docs, personas, cfg.bm25_k1, cfg.bm25_b,
            analyzer=tokenize_keep_stopwords,
        )
        # Substring matching, as an index rather than a scan: char n-grams turn
        # "facilityfact" — which spans a separator and so is nobody's token —
        # into ordinary terms to look up.
        self.name_ngram = (
            NgramIndex(ids, name_alias_docs, personas, cfg.ngram_range)
            if cfg.name_ngram_weight > 0.0 else None
        )

        # A collection loaded before `columns` was written to metadata yields an
        # all-empty corpus, which CountVectorizer rejects outright ("empty
        # vocabulary"). That would raise through the whole index build and be
        # swallowed by warm()'s except, degrading search to vector-only with no
        # visible cause — so say it plainly and carry on without the retriever.
        self.column_bm25: Optional[Bm25Index] = None
        if cfg.column_weight > 0.0:
            if any(column_docs):
                self.column_bm25 = Bm25Index(
                    ids, column_docs, personas, cfg.bm25_k1, cfg.bm25_b,
                )
            elif ids:
                logger.warning(
                    "[keyword_index] No column text in the catalog metadata — the "
                    "column retriever is disabled. Re-run the loader to populate it.",
                )
    def search_with_breakdown(
        self,
        query: str,
        top_n: int = 10,
        persona_id: Optional[int] = None,
        cfg=None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Return per-retriever hits plus the combined lexical result list."""
        cfg = cfg or get_search_config()
        candidate_n = max(top_n * cfg.candidate_mult, cfg.candidate_min)

        bm25_hits = self.bm25.search(query, top_n=candidate_n, persona_id=persona_id)
        if self.ngram is not None and cfg.ngram_weight > 0.0:
            ng_hits = self.ngram.search(query, top_n=candidate_n, persona_id=persona_id)
            general_hits = fuse_rrf(
                [h[0] for h in bm25_hits],
                [h[0] for h in ng_hits],
                k=cfg.rrf_k,
                w_a=1.0,
                w_b=cfg.ngram_weight,
            )
        else:
            ng_hits = []
            general_hits = bm25_hits

        # Column retriever. Fused at the general level because it is content evidence
        # like BM25 over descriptions, not name evidence — the name/alias retriever
        # keeps its own precedence below.
        column_hits: list[tuple[str, float]] = []
        if self.column_bm25 is not None:
            column_hits = self.column_bm25.search(
                query, top_n=candidate_n, persona_id=persona_id,
            )
        general_hits = _fuse_or_keep(general_hits, column_hits, cfg.column_weight, cfg.rrf_k)

        name_alias_hits = self.name_alias_bm25.search(
            query,
            top_n=candidate_n,
            persona_id=persona_id,
        )

        name_ngram_hits: list[tuple[str, float]] = []
        if self.name_ngram is not None:
            name_ngram_hits = self.name_ngram.search(
                query, top_n=candidate_n, persona_id=persona_id,
            )

        name_alias_hits = _fuse_or_keep(
            name_alias_hits, name_ngram_hits, cfg.name_ngram_weight, cfg.rrf_k,
        )
        keyword_hits = _fuse_or_keep(
            general_hits, name_alias_hits, cfg.name_alias_weight, cfg.rrf_k,
        )

        return {
            "keyword_hits": keyword_hits[:top_n],
            "general_keyword_hits": general_hits[:top_n],
            "name_alias_hits": name_alias_hits[:top_n],
            "name_ngram_hits": name_ngram_hits[:top_n],
            "bm25_hits": bm25_hits[:top_n],
            "ngram_hits": ng_hits[:top_n],
            "column_hits": column_hits[:top_n],
        }

    def search(
        self,
        query: str,
        top_n: int = 10,
        persona_id: Optional[int] = None,
        cfg=None,
    ) -> list[tuple[str, float]]:
        """Return combined lexical hits (general keyword + name/alias retriever)."""
        return self.search_with_breakdown(
            query,
            top_n=top_n,
            persona_id=persona_id,
            cfg=cfg,
        )["keyword_hits"]


# ----------------------------------------------------------------------------
# Fingerprint / cache helpers
# ----------------------------------------------------------------------------

def _index_fingerprint(collection_data: dict, cfg) -> str:
    """SHA-256 over everything the cached matrices were built from.

    That includes the analyzer's *data*, not just the numeric config: ``_ABBREVIATIONS``
    and ``_STOPWORDS`` decide what every document tokenizes to. Leave them out
    and adding ``"fac": "facility"`` to the glossary is silently inert — the
    matrix built under the old vocabulary keeps being served against queries
    tokenized under the new one, until something unrelated changes the catalog.

    Query-time-only weights are deliberately *not* here. They shape no matrix,
    so including them only forces a full rebuild for a knob that costs nothing.
    """
    ids = collection_data.get("ids") or []
    metas = collection_data.get("metadatas") or []
    index_cfg = {
        "cache_format": _CFG_CACHE_FORMAT,
        "table_name_boost_version": _CFG_TABLE_NAME_BOOST_VERSION,
        "fields": sorted(cfg.keyword_fields),
        "k1": cfg.bm25_k1,
        "b": cfg.bm25_b,
        # Structural: these decide whether the n-gram and column retrievers are built.
        "ngram_weight": cfg.ngram_weight,
        "column_weight": cfg.column_weight,
        "name_ngram_weight": cfg.name_ngram_weight,
        "ngram_range": list(cfg.ngram_range),
        "abbreviations": sorted(_ABBREVIATIONS.items()),
        "stopwords": sorted(_STOPWORDS),
    }
    h = hashlib.sha256()
    h.update(json.dumps(index_cfg, sort_keys=True).encode())
    # Sort by id for determinism
    for _id, meta in sorted(zip(ids, metas), key=lambda x: x[0]):
        h.update(_id.encode())
        h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()


def _cache_path() -> Path:
    # Same default as chroma_db.py when the env var is unset.
    storage = os.getenv("VECTORDB_STORAGE_PATH") or str(
        Path(__file__).resolve().parents[2] / "data"
    )
    return Path(storage) / "keyword_index" / "table_catalog.pkl"


def _load_from_cache(fingerprint: str) -> Optional[KeywordIndex]:
    if not _CFG_INDEX_CACHE:
        return None
    p = _cache_path()
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("fingerprint") == fingerprint:
            logger.info("[keyword_index] Loaded from cache: %s", p)
            return cached["index"]
        logger.info("[keyword_index] Cache fingerprint mismatch — rebuilding.")
    except Exception as exc:
        logger.warning("[keyword_index] Failed to load cache (%s) — rebuilding.", exc)
    return None


def _save_to_cache(index: KeywordIndex, fingerprint: str) -> None:
    if not _CFG_INDEX_CACHE:
        return
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"fingerprint": fingerprint, "index": index}, fh)
        tmp.replace(p)
        logger.info("[keyword_index] Index cached at: %s", p)
    except Exception as exc:
        logger.warning("[keyword_index] Could not write cache (%s).", exc)


# ----------------------------------------------------------------------------
# Build from ChromaDB collection
# ----------------------------------------------------------------------------

def build_from_collection(collection, cfg=None) -> KeywordIndex:
    """Fetch all rows from a ChromaDB VectorCollection and build a KeywordIndex."""
    cfg = cfg or get_search_config()
    logger.info("[keyword_index] Fetching all rows from collection for keyword index build ...")
    data = collection.get(include=["metadatas"])
    ids: list[str] = data.get("ids") or []
    metas: list[dict] = data.get("metadatas") or []
    logger.info("[keyword_index] Fetched %d rows.", len(ids))

    fingerprint = _index_fingerprint(data, cfg)
    cached = _load_from_cache(fingerprint)
    if cached is not None:
        return cached

    docs = [_build_keyword_doc(m, set(cfg.keyword_fields)) for m in metas]
    name_alias_docs = [_build_name_alias_doc(m) for m in metas]
    column_docs = [_build_column_doc(m) for m in metas]
    personas = [_persona_ids_of(m) for m in metas]

    logger.info("[keyword_index] Building BM25 index over %d documents ...", len(ids))
    index = KeywordIndex(
        ids, docs, name_alias_docs, column_docs, personas, cfg,
    )
    logger.info("[keyword_index] BM25 index built. Vocab size: %d", len(index.bm25.vocab))

    _save_to_cache(index, fingerprint)
    return index


# ----------------------------------------------------------------------------
# KeywordIndexService — singleton with lock / lazy / warm / invalidate
# ----------------------------------------------------------------------------

class KeywordIndexService:
    """Thread-safe lazy singleton wrapping a ``KeywordIndex``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: Optional[KeywordIndex] = None

    def get(self) -> Optional[KeywordIndex]:
        """Return the current index (may be None if not yet built)."""
        return self._index

    def warm(self, collection=None, cfg=None) -> bool:
        """Build/load the index if not already built.

        ``collection`` may be passed directly; otherwise the service opens
        the ``table_catalog`` collection itself.

        Returns True on success, False on failure.
        """
        cfg = cfg or get_search_config()
        if not any((cfg.keyword_weight, cfg.ngram_weight, cfg.name_alias_weight, cfg.column_weight)):
            logger.info("[keyword_index] Every lexical weight is zero — skipping warm.")
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
                logger.info("[keyword_index] warm() complete.")
                return True
            except Exception as exc:
                logger.error("[keyword_index] warm() failed: %s", exc, exc_info=True)
                return False

    def invalidate(self) -> None:
        """Drop the in-memory index (cache file is NOT deleted)."""
        with self._lock:
            self._index = None
        logger.info("[keyword_index] In-memory index invalidated.")

    def _open_collection(self):
        try:
            from app.rag.vector_store import get_collection
            from app.rag.embedding_vertex import get_vectordb_embedding_fn
            return get_collection(
                COLLECTION_NAME,
                get_vectordb_embedding_fn(task="RETRIEVAL_DOCUMENT"),
            )
        except Exception as exc:
            logger.error("[keyword_index] Could not open collection: %s", exc)
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
        print('Usage: python -m app.rag.keyword_index "query text"')
        print('       python -m app.rag.keyword_index --dump-vocab')
        return

    svc = get_keyword_index_service()
    svc.warm()
    idx = svc.get()
    if idx is None:
        print("Index not available.")
        return

    if "--dump-vocab" in args:
        for term in sorted(idx.bm25.vocab.keys()):
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

