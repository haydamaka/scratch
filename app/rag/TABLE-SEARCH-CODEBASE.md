# Table search — cheatsheet

Entry point: `hybrid_search.search(query, top_n, persona_id, **overrides)`.
Everything below is reached from there.

## 0. Glossary

**Dense vs sparse.** *Dense* = embed query and document into vectors, rank by cosine
distance; matches meaning, not words. *Sparse* = score on the words themselves
(one dimension per vocabulary term, mostly zeros). We run both and fuse.

**TF-IDF** — the baseline sparse weight for a term in a document:
`tf × log(N / df)`. *Term frequency* (how often the term occurs here) × *inverse document
frequency* (how rare it is across the catalog). Rare terms carry the signal; terms in every
document carry none. Rank = cosine similarity between the query's TF-IDF vector and each
document's.

**BM25** — TF-IDF's successor, and what we use for word-level matching. Two fixes:
- *Saturation* (`k1`, default 1.2) — the 10th occurrence of "amount" adds far less than the
  2nd. Raw `tf` grows without bound; BM25's flattens out.
- *Length normalisation* (`b`, default 0.4) — a long description shouldn't out-score a short
  one just by containing more words. `b=0` ignores length, `b=1` fully normalises; we sit
  low because our documents are short and uneven.

  `score = idf × tf(k1+1) / (tf + k1(1 − b + b·len/avglen))`

**Char n-gram (TF-IDF)** — same TF-IDF maths, but the "terms" are overlapping character
runs of length 3–5 (`char_wb`, so they don't cross word boundaries) instead of words. That
is what lets `facilityfact` match `facility_fact`, and what bridges abbreviations no
synonym map covers. Fuzzy by construction — hence its low weight (0.3).

**RRF (Reciprocal Rank Fusion)** — how we merge retrievers. Each contributes
`weight / (k + rank)` per document (`k=60`), summed. Uses only *rank*, never score, so BM25
scores and cosine distances never have to be made comparable — which is exactly why
score-blending (`fuse_weighted`) was dropped.

## 1. Retrievers and how they combine

```
query ─┬─ expand_lexical_query ─→ BM25(desc) ─┬RRF─ general ─┐
       │                          ngram ──────┘              │
       │                          columns ────────RRF────────┤
       │                          name/alias ─┬RRF─ name ────┴RRF─ keyword_hits ─┐
       │                          name-ngram ─┘                                  ├RRF─ ranked
       └─ embed ────────────────→ dense (Chroma) ────────────────────────────────┘        │
                                                                                          ↓
                              hits ← assemble ← back-fill ← ER expansion (er_filter) ←─────┘
```

Six retrievers. All fusion is RRF (`weight / (k + rank)`), never score-blending —
`fuse_weighted` was removed as unstable.

| # | Retriever | Algorithm | Corpus | Where it fuses |
|---|---|---|---|---|
| 1 | **dense** | embedding cosine | embedded doc: `TABLE:` / `ALIASES:` / `DESCRIPTION:` (no columns) | top level, vs `keyword_hits` |
| 2 | **bm25** | BM25, `tokenize` | name + alias + domain + description + rules | → `general` |
| 3 | **ngram** | char TF-IDF 3–5 | same corpus as bm25 | → `general` |
| 4 | **columns** | BM25, `tokenize` | `metadata["columns"]`, untruncated | → `general` |
| 5 | **name_alias** | BM25, `tokenize_keep_stopwords` | table name + aliases | → `name` |
| 6 | **name_ngram** | char TF-IDF 3–5 | table name + aliases | → `name` |

3 BM25 + 2 char-n-gram TF-IDF + 1 dense. The two n-gram retrievers are *not* BM25 —
`NgramIndex` is a plain `TfidfVectorizer`.

### "Where it fuses" — the fusion tree

Fusion is **not** one flat RRF over six lists. It is a tree, built bottom-up in
`KeywordIndex.search_with_breakdown` and finished in `hybrid_search._combine_rankings`:

```
                                    w_a      w_b
bm25       ⊕ ngram       → general  1.0   ngram_weight        (0.3)
general    ⊕ columns     → general  1.0   column_weight       (1.0)
name_alias ⊕ name_ngram  → name     1.0   name_ngram_weight   (0.3)
general    ⊕ name        → keyword  1.0   name_alias_weight   (1.0)
dense      ⊕ keyword     → ranked   vector_weight  keyword_weight
```

Two grouping decisions drive it:

- **`general` = what the table is about.** Description, domain, rules, column names — content
  evidence, scored by rarity. `columns` joins here rather than at `name` for that reason:
  a column list describes content, it is not a name.
- **`name` = what the table is called.** Name and aliases only, with stopwords kept so
  `fact`, `dim`, `amount` survive and IDF weights them. Kept separate so an exact-ish name
  match isn't diluted by a long description that happens to share vocabulary.

Then the two meet, and only then does the whole sparse side meet dense.

**The consequence to remember: every weight is local, not global.** `_fuse_or_keep` always
passes `w_a=1.0` for the accumulating side, so a weight means "relative to whatever is
already in this node". `ngram_weight=0.3` is 0.3 *against bm25*; `name_alias_weight=1.0` is
parity *against `general`, which by then already contains three retrievers*. Doubling
`column_weight` does not double the column retriever's influence on the final ranking — it
doubles it inside `general`, which is then re-weighted twice more on the way up. To reason
about end-to-end influence you have to walk the tree.

Each level also degrades independently: a zero weight or an empty list leaves the
accumulator untouched, rather than fusing against nothing and replacing real scores with
RRF pseudo-scores in the same order.

Around that ranking:
- **Query expansion** — `expand_lexical_query` appends glossary synonyms. Sparse
  retrievers only; the dense retriever and the exact-match pin see the raw query.
- **Abbreviations** — `_ABBREVIATIONS` in `keyword_index`, symmetric (docs *and* queries),
  applied inside the tokenizer.
- **Exact-match pin** — `promote_exact_match` forces an exact name/alias hit to rank 1 in
  both lists, before fusion.
- **ER expansion** — `er_filter.select_with_related_tables` reads `ranked` `er_candidates_n`
  deep, keeps the top `er_anchor_n` as **anchors**, and fills the rest only with tables the
  relationship graph joins to those. Scored on rank + dim-ness + connection count +
  edge confidence — *not* on text score, since text score is what failed to surface them.
- **Persona filter** — `persona_filter()` → `{persona_id_<pid>: 1}`. Use the flags, not the
  scalar `persona_id` (that holds only the first id).

## 2. The knobs — all in `hybrid_search.SearchConfig`

One frozen dataclass, one place. Defaults resolve on **first use**, not import, so `.env`
loaded by `bootstrap_standalone()` is honoured. Override per call:

```python
search("utilised amount", top_n=30, column_weight=2.0, er_candidates_n=0)
```

### Depth — how many rows each stage handles

| Field | Default | Env | What it does |
|---|---|---|---|
| `top_n` | 5 | `SEARCH_TOP_N` | How many tables `search()` returns, when the caller doesn't pass one. |
| `candidate_mult` | 5 | `KEYWORD_CANDIDATE_MULT` | Fetch depth multiplier. Each retriever pulls `top_n × mult` so fusion has a pool to work with — a retriever that only ever returned `top_n` could never rescue a table the other ranked 30th. |
| `candidate_min` | 50 | `KEYWORD_CANDIDATE_MIN` | Floor on that depth, so a `top_n=3` call still fuses over a meaningful pool instead of 15 rows. |

Fetch depth = `candidate_depth()` = `max(top_n × mult, min, er_candidates_n if ER active)`.
The ER pool only raises the floor when expansion can actually happen (`top_n > er_anchor_n`);
otherwise there are no slots to fill and the deeper fetch is waste.

### Retriever weights — query time, effective on the next call

| Field | Default | Env | What it does |
|---|---|---|---|
| `rrf_k` | 60 | `KEYWORD_RRF_K` | RRF's rank-damping constant, in `weight/(k+rank)`. **Low** `k` (~10) makes rank 1 dominate — trust each retriever's top hit. **High** `k` (~100) flattens the curve, so agreement across retrievers matters more than any one's ordering. |
| `vector_weight` | 1.0 | `VECTOR_WEIGHT` | Weight of the dense retriever in the final fusion. |
| `keyword_weight` | 1.0 | `KEYWORD_WEIGHT` | Weight of the whole sparse side (`keyword_hits`) against dense. `0.0` skips all sparse retrieval — the vector-only escape hatch. |
| `ngram_weight` | 0.3 | `NGRAM_WEIGHT` | Char-n-gram retriever against BM25 inside `general`. Low because it is fuzzy by construction: it matches substrings, so it will always surface near-misses that BM25 correctly rejected. |
| `column_weight` | 1.0 | `KEYWORD_COLUMN_WEIGHT` | Column-name retriever, also into `general`. Fused there rather than into `name` because column names are *content* evidence, like a description — not name evidence. |
| `name_alias_weight` | 1.0 | `KEYWORD_NAME_ALIAS_WEIGHT` | The name/alias side against `general` in the last sparse fusion. Raise it when questions tend to name tables; lower it when they describe them. |
| `name_ngram_weight` | 0.3 | `NAME_NGRAM_WEIGHT` | Char-n-grams over names/aliases against name/alias BM25. This is what matches `facilityfact` to `facility_fact` — a span no tokenizer produces. |

Setting one of `ngram_weight`, `column_weight`, `name_ngram_weight` to `0.0` also skips
*building* that retriever, so it's a build-time knob too.

### Index build — changing any of these rebuilds the keyword index

They are covered by the cache fingerprint, so editing one evicts the pickle on the next
build. Until that rebuild happens, the value is inert.

| Field | Default | Env | What it does |
|---|---|---|---|
| `bm25_k1` | 1.2 | `BM25_K1` | Term-frequency saturation. Higher = repeats keep counting; lower = one occurrence is nearly as good as five. `0` reduces BM25 to presence/absence. |
| `bm25_b` | 0.4 | `BM25_B` | Length normalisation, `0`–`1`. Below the 0.75 default on purpose: our documents are short and very uneven (a rich description vs a bare name), and full normalisation over-rewards the sparse ones. |
| `ngram_range` | `(3, 5)` | `NGRAM_RANGE` | Character-n-gram lengths. Wider = more recall and a much larger vocabulary; `(3,5)` spans a short abbreviation up to a word stem. |
| `keyword_fields` | `name,alias,domain,description,rules` | `KEYWORD_FIELDS` | Which metadata fields go into the BM25 document. Dropping `rules` loses business vocabulary ("fronting", "intercompany") that is in no other indexed field and is not embedded either. |

### ER expansion — the relationship-graph step

Env name is the uppercased field name for all of these (`ER_CANDIDATES_N`, …); only the
fields listed in `_ENV_NAMES` differ from that rule.

| Field | Default | What it does |
|---|---|---|
| `er_candidates_n` | 80 | How deep into the fused ranking to look for graph-connected tables. `0` disables the step entirely. |
| `er_anchor_n` | 10 | How many top-ranked tables are kept unconditionally — the **anchors**. Everything added below them must connect to one of these. |
| `er_related_n` | 20 | Cap on tables added below the anchors. Actual slots = `min(this, top_n − er_anchor_n)`. |
| `er_weight_rank` | 1.0 | Weight on the candidate's own text rank — the one signal that already failed to surface it, hence the three boosts below. |
| `er_weight_dimension` | 0.5 | Boost for a dimension table. A fact table among the anchors joins its dimensions, and they carry the codes a question filters on: cheap to include, expensive to omit. |
| `er_weight_connections` | 0.3 | Boost for joining *several* anchors rather than one — a real dependency of the query, not something hanging off a single table. |
| `er_weight_confidence` | 0.2 | Boost by the graph's own edge confidence, so an inferred edge admits a table less readily than a declared key. |
| `er_connections_for_full_score` | 3 | Connection count at which `er_weight_connections` is fully earned. |

Each term is normalised to `[0,1]`, so a weight is literally "how many rank places this is
worth". The three boosts sum to `er_weight_rank` on purpose: everything going right for a
candidate can at most fully compensate for being ranked last, never more.

### Not in SearchConfig

| Env | Where | What it does |
|---|---|---|
| `KEYWORD_INDEX_CACHE` | `keyword_index.py` | `0` disables the pickle cache — rebuild on every warm. |
| `_CFG_CACHE_FORMAT` / `_CFG_TABLE_NAME_BOOST_VERSION` | `keyword_index.py` | Constants, not env. Bump when a change to the pickled classes would make old caches unreadable. |
| `VECTORDB_RELOAD` | `vectordb_loader.py` | Gates the catalog load at startup; unset means serve the persisted store. |
| `VECTORDB_STORAGE_PATH` | `keyword_index._cache_path` | Where the index pickle lives (`<path>/keyword_index/table_catalog.pkl`). |

## 3. Index lifecycle — when the sparse indices get built

Every sparse retriever is a **fitted, materialised index**, not a scan. Nothing matches at
query time without one having been built first.

| | Built by | Holds |
|---|---|---|
| `Bm25Index` ×3 | `CountVectorizer.fit_transform` → BM25 reweight | vocabulary `{term: col}` + CSC weight matrix |
| `NgramIndex` ×2 | `TfidfVectorizer(char_wb, 3–5).fit_transform` | fitted vectorizer (n-gram vocabulary **+ IDF vector**) + transposed CSC matrix |

The n-gram retrievers are the *heaviest* of the five, not the lightest: every distinct 3-to-5
character run in the corpus becomes a vocabulary entry, so the term space is far larger than
the word-token one. Query time is `vec.transform([query])` then one sparse mat-mul — cheap,
but only because the fit already happened.

### The three build triggers

All funnel through `KeywordIndexService.warm()`, which returns immediately if
`self._index is not None`:

1. **Startup** — `vectordb_loader._run()` calls `warm()` after a successful load, and also
   when the load was skipped and we are serving the persisted store. This is the intended path.
2. **Lazily, on the first search** — `hybrid_search._sparse_retrieve()` calls `warm()` on
   *every* query. Normally a no-op; if the startup warm failed or never ran (a CLI, a test,
   a worker that skipped the loader), the first query pays for the build instead of the
   search silently degrading to vector-only.
3. **After a catalog reload** — `table_info_loader.load_from_api()` calls `invalidate()` at
   the end, which drops the in-memory index (**not** the pickle). The next `warm()` re-enters
   the build path.

### There is no incremental build. Only two outcomes.

`build_from_collection()` always does the same thing: fetch **all** rows →
`_index_fingerprint(rows, cfg)` → try cache → fit → save cache. Which of two things happens:

- **Cache hit** (fingerprint identical): unpickle the whole index. Milliseconds. Nothing is
  fitted.
- **Cache miss** (anything differs): all five matrices fitted from scratch over the whole
  catalog. There is no third path — no append, no partial update, no per-row patch.

The fingerprint is a SHA-256 over the build-time config plus **every** id and metadata dict,
sorted. One table's description edited, one row added, one `bm25_b` changed → new hash →
full re-fit. `KeywordIndex`, `Bm25Index` and `NgramIndex` expose no `add`/`update`/`delete`;
they are constructed once and are query-only thereafter.

**Why incremental isn't just missing work.** Both index types bake *corpus-wide* statistics
into every stored weight:

- BM25 stores `idf × tf(k1+1)/(tf + k1(1−b+b·len/avgdl))` per cell. `idf` needs the document
  frequency of that term across the whole corpus, and `avgdl` is the mean document length.
  Add one table and both shift, so every previously-computed cell is stale — not just the
  new row's.
- `TfidfVectorizer` is the same for `idf_`, and worse for vocabulary: it is fixed at `fit`
  time, so a new document's unseen n-grams have no column to go in.

Appending a row is therefore not a local operation. Incremental BM25 is possible in
principle (store raw `tf` and `df` counters, recompute weights lazily), but that is a
different data structure from a pre-weighted CSC matrix — a deliberate trade of update cost
for query speed.

So: `invalidate()` is cheap on its own (it forces a re-entry, not a re-fit — a reload that
changed nothing still hits the cache). What costs is the fingerprint moving, and when it
moves you pay for the entire catalog.

**Gotcha:** `upsert_one()` does *not* invalidate — only `load_from_api()` does. An ad-hoc
single-table upsert leaves a running process searching a stale keyword index until something
else invalidates or the process restarts. It self-heals on restart, because the fingerprint
will have changed by then.

Cache file: `$VECTORDB_STORAGE_PATH/keyword_index/table_catalog.pkl`, written atomically via
`.tmp` + `replace`. `KEYWORD_INDEX_CACHE=0` disables it, which turns every cold start into a
full fit.

## 4. File / function map

### `hybrid_search.py` — the pipeline and the config
| | |
|---|---|
| `SearchConfig` / `from_env` / `with_overrides` | every knob; `_ENV_NAMES` maps field → env var |
| `get_search_config()` | process-wide singleton, resolved once |
| `search()` | the whole pipeline, returns hit dicts |
| `candidate_depth()` | per-retriever fetch depth |
| `dense_retrieve()` | Chroma query → `{id: (meta, doc, distance)}` |
| `_sparse_retrieve()` | glossary-expands, calls `KeywordIndex.search` |
| `_combine_rankings()` | RRF when both ran, else whichever did |
| `_add_related_tables()` | calls `er_filter` |
| `_fetch_missing_records()` | back-fills metadata for keyword-only hits |
| `fuse_rrf()` / `promote_exact_match()` | fusion primitives (pure) |

### `keyword_index.py` — the sparse retrievers
| | |
|---|---|
| `_ABBREVIATIONS` | `cpty` → `counterparty`; symmetric, docs + queries |
| `_split_identifier()` | `schema.table` / camelCase / underscore → sub-tokens |
| `tokenize()` / `tokenize_keep_stopwords()` | the two analyzers (module-level so they pickle) |
| `_build_keyword_doc()` / `_build_column_doc()` / `_build_name_alias_doc()` | the three corpora |
| `Bm25Index` | Okapi BM25, Lucene IDF, CSC weight matrix |
| `NgramIndex` | char-`wb` TF-IDF |
| `_PersonaMasks` | per-persona row masks |
| `KeywordIndex.search_with_breakdown()` | **runs and fuses all five sparse retrievers** — read this one |
| `KeywordIndex.search()` | → `["keyword_hits"]` |
| `build_from_collection()` | reads Chroma, builds, caches |
| `_index_fingerprint()` / `_load_from_cache()` / `_save_to_cache()` | SHA-256 over cfg + rows → pickle |
| `KeywordIndexService` | thread-safe lazy singleton: `get` / `warm` / `invalidate` |

### `vector_store.py` — provider-agnostic store access
`get_collection()` / `get_or_create_collection()` / `store_location()`, and
`persona_filter()` → `{persona_id_<pid>: {"$eq": 1}}` — the read side of the flags the
loaders write. Chroma filter dialect; `milvus_db._translate_where` converts it.

### `query_prep.py` — query-side only
`GLOSSARY` (one-directional, `utilized` → `outstanding, drawn, used`) and
`expand_lexical_query()`, which **appends** rather than substitutes. Never applied to
documents or to the dense retriever.

### `er_filter.py` — relationship-graph expansion
`select_with_related_tables()` (the entry point; returns `(direct, related, used_graph)` —
`used_graph=False` means the fallback took tables on text rank and they must not be
reported as ER-expanded), `_candidate_score()`, `_connected_tables()`, `is_dim_table()`.

### `table_info_loader.py` — what gets indexed
`_build_document()` (embedded text; **columns deliberately excluded**),
`_build_metadata()` (`columns` untruncated for BM25, `persona_ids` + `persona_id_<pid>` flags),
`load_from_api()` / `load_from_json()` / `upsert_one()`.

### `table_info_search.py` — catalog-facing surface
`get_search_hit_info()` is a thin wrapper over `hybrid_search.search` (kept under that name
because the endpoint, the CLI and both eval harnesses call it). The rest is Chroma lookups
and diagnostics: `_retriever_hits()` / `get_retriever_hits_detailed()`,
`get_query_distances_for_tables()`, `get_table_records()`, `print_*`.

### `vectordb_loader.py`
Background load + readiness gate; warms the keyword index after a successful load.

### `test/validate_table_retrieval.py`
Eval harness — per-question `<n>-success.txt` / `<n>-failure.txt` with per-retriever hits,
metrics and ER distance.

## 5. Libraries

Four, plus the embedder. The ranking logic itself — BM25, RRF, the ER scoring — is ours.

| Library | Version | Used for |
|---|---|---|
| **scikit-learn** | 1.5.1 | `CountVectorizer` — term-frequency matrix that `Bm25Index` reweights. `TfidfVectorizer(analyzer="char_wb")` — *is* both n-gram retrievers. `ENGLISH_STOP_WORDS` |
| **scipy** | 1.17.1 | `csc_matrix` — the BM25 weight matrix. Column-sparse, so a query slices its terms' columns and sums. Direct dependency, not transitive — see the note in `requirements.txt` |
| **numpy** | 1.26.4 | score vectors, persona row masks, `argpartition` top-k |
| **chromadb** | 0.5.0 | the vector store: persistence, the dense query, metadata `where` filters |
| **Vertex AI** `text-embedding-005` | via the platform SDK | the embeddings, behind a managed token service. Asymmetric — `RETRIEVAL_DOCUMENT` at index time, `RETRIEVAL_QUERY` at query time |

Deliberately **not** used:
- **No `rank_bm25`.** BM25 is ~40 lines here (`Bm25Index.__init__`) and vectorised; a
  pure-Python per-query loop over documents would not hold at catalog scale.
- **No `nltk` / `spacy`.** Tokenisation is identifier-aware (`schema.table`, camelCase,
  digit boundaries) — linguistic tokenisers get those wrong. Stopwords are sklearn's list
  plus `_DOMAIN_STOPWORDS`.
- **No cross-encoder / reranker.** RRF over six retrievers plus the ER step is the whole
  ranking; nothing loads a model at query time except the embedder.

## 6. CLI

```bash
python -m app.rag.table_info_search "query text"
python -m app.rag.keyword_index "query text"
python -m app.rag.keyword_index --dump-vocab
python -m app.rag.table_info_loader          # rebuild the catalog
```
