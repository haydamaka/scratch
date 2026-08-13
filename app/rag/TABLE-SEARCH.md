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
- *Saturation* (`k1`, default 1.2) — the 10th occurrence of "status" adds far less than the
  2nd. Raw `tf` grows without bound; BM25's flattens out.
- *Length normalisation* (`b`, default 0.4) — a long description shouldn't out-score a short
  one just by containing more words. `b=0` ignores length, `b=1` fully normalises; we sit
  low because our documents are short and uneven.

  `score = idf × tf(k1+1) / (tf + k1(1 − b + b·len/avglen))`

**Char n-gram (TF-IDF)** — same TF-IDF maths, but the "terms" are overlapping character
runs of length 3–5 (`char_wb`, so they don't cross word boundaries) instead of words. That
is what lets `entityfact` match `entity_fact`, and what bridges abbreviations no
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
  `fact`, `dim`, `code` survive and IDF weights them. Kept separate so an exact-ish name
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
search("some query text", top_n=30, column_weight=2.0, er_candidates_n=0)
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
| `name_ngram_weight` | 0.3 | `NAME_NGRAM_WEIGHT` | Char-n-grams over names/aliases against name/alias BM25. This is what matches `entityfact` to `entity_fact` — a span no tokenizer produces. |

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
| `keyword_fields` | `name,alias,domain,description,rules` | `KEYWORD_FIELDS` | Which metadata fields go into the BM25 document. Dropping `rules` loses the business jargon that lives only there — vocabulary in no other indexed field, and not embedded either. |

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

## 3. The hardcoded term maps

Two of them, at different points in the pipeline and for different reasons.

| Map | Where | Applied to | Takes effect |
|---|---|---|---|
| `_ABBREVIATIONS` — ~40 pairs, `amt`→`amount`, `txn`→`transaction`, `ccy`→`currency` | `keyword_index.py:61`, used inside `_split_and_normalize`, which both tokenizers call | documents **and** queries, symmetric by construction | index rebuild — it is in the cache fingerprint, so editing it evicts the pickle |
| `GLOSSARY` — a handful of domain phrases, `utilized`→`outstanding, drawn, used, available` | `query_prep.py:25`, applied by `expand_lexical_query` | the query only, and only on the sparse path: dense and the exact-match pin see the raw query | next call |

**Why the abbreviations are required.** BM25 scores exact tokens, and `amt` and `amount`
share none. Questions arrive as prose; the catalog's names and columns are abbreviated
snake_case. Without normalising both sides to one surface there is no term in common, and
no weight can rescue a match that does not exist. The char-n-gram retriever bridges part of
this (`entityfact` → `entity_fact`) but fuzzily, which is why it sits at 0.3.

**Why the glossary is required.** It covers what no tokenizer can reach: the question and
the catalog use *different words* for the same thing — "utilized amount" against a
description written as "outstanding direct, contingent, and unused commitment amounts".
That is semantics, not spelling. Expansions are appended rather than substituted, since the
user's own wording may be exactly what the catalog used; each entry was added from an
observed miss, and the comment above it names the table it was added for.

## 4. Libraries

Four, plus the embedder. The ranking logic itself — BM25, RRF, the ER scoring — is ours.

| Library | Version | Used for |
|---|---|---|
| **scikit-learn** | 1.5.1 | `CountVectorizer` — term-frequency matrix that `Bm25Index` reweights. `TfidfVectorizer(analyzer="char_wb")` — *is* both n-gram retrievers. `ENGLISH_STOP_WORDS` |
| **scipy** | 1.17.1 | `csc_matrix` — the BM25 weight matrix. Column-sparse, so a query slices its terms' columns and sums. Direct dependency, not transitive — see the note in `requirements.txt` |
| **numpy** | 1.26.4 | score vectors, persona row masks, `argpartition` top-k |
| **chromadb** | 0.5.0 | the vector store: persistence, the dense query, metadata `where` filters |
| **Vertex AI** `text-embedding-005` | via the platform SDK | the embeddings, behind a managed token service. Asymmetric — `RETRIEVAL_DOCUMENT` at index time, `RETRIEVAL_QUERY` at query time |
