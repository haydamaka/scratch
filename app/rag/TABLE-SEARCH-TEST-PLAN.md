# Table search — evaluation guide

How to measure table retrieval on the machine that has the data, and what to bring
back so the configuration can be chosen from evidence rather than intuition.

Two harnesses, same CSVs, same loader:

| | Answers | Writes |
|---|---|---|
| `test/table_retrieval_analyser.py` | *which configuration is better* | one `summary.json` per run |
| `test/validate_table_retrieval.py` | *why did this question fail* | per-question `<n>-success.txt` / `<n>-failure.txt` |

Sweep with the analyser; reach for the validator on the questions the analyser flags.
Knobs: see `TABLE-SEARCH-CODEBASE.md` §2.

---

## 1. Prerequisites

**Data.** Put the eval CSVs in `app/rag/test/data/*.csv`. Every `*.csv` there is read
and the rows are merged, deduplicated on the question text. Columns are matched **by
name**, case-insensitive; extra columns are ignored:

| Column | Required | Notes |
|---|---|---|
| `QUESTION` | yes | rows with an empty one are skipped |
| `EXPECTED_TABLE` | yes | the ground truth. One or more `schema.table`, comma- or newline-separated |
| `QUESTION_ID` | no | only names the report file; falls back to `<file stem>-<row>` |
| `CATEGORY` | no | carried into the report and the summary |
| `GROUND_TRUTH_SQL` | no | read but not scored — the harness matches table names, not SQL |

**Services.** The catalog must be loaded and the keyword index built:

```bash
ENV=uat .venv/bin/python -m app.rag.vectordb_loader     # loads both catalogs, warms the index
```

Requires the embedder to be reachable — this is the one step that needs
the corporate network. ER expansion additionally needs the relationship graph
(`RELATIONSHIPS_JSON` or the TI service); without it the harness still runs and
`used_graph` goes false, so ER contributes nothing rather than crashing.

## 2. Smoke test before the real run

```bash
ENV=uat .venv/bin/python -m app.rag.table_info_search "utilised amount"   # search works
ENV=uat .venv/bin/python -m app.rag.keyword_index --dump-vocab | wc -l    # index is built
ENV=uat .venv/bin/python -m app.rag.test.table_retrieval_analyser --limit 5
```

The third should print `Loaded 5 question row(s)` and finish with a `Done. questions=5 …`
line. If it says `Loaded 0`, the CSVs are not where the harness looks or lack the two
required columns.

## 3. Baseline

```bash
ENV=uat .venv/bin/python -m app.rag.test.table_retrieval_analyser \
    --top-n 30 --label baseline --out ./results/baseline/summary.json
```

Then, once — the per-question diagnostics, for the questions the numbers flag:

```bash
ENV=uat .venv/bin/python -m app.rag.test.validate_table_retrieval \
    --top-n 30 --out-dir ./results/baseline/reports
```

Read the analyser's final three log lines first. They answer "is tuning even the right lever":

```
strict success 42.0% | micro recall 0.71 | ceiling (union of retrievers) 0.86
                     | 12 GT via ER | 9 GT unreachable by any retriever
recall@k (final list): @1=0.31 @3=0.52 @5=0.61 @10=0.68 @20=0.71 @30=0.71
per-retriever recall @top_n: vector=0.62 keyword=0.70 name_alias=0.44
```

- **`micro recall` vs `ceiling`** — the gap is what weight and depth tuning can win.
  Some retriever already found those tables; fusion just didn't rank them high enough.
- **`unreachable`** — ground-truth tables that *no* retriever surfaced at any depth.
  No weight fixes these. They need loader work (indexed fields, descriptions, aliases,
  the glossary) or the relationship graph. Count them before tuning anything.
- **`recall@k` flattening** — if `@10` ≈ `@30`, raising `top_n` is not the lever.

## 4. What to send back

For each run, the whole `summary.json`. That is enough to reconstruct every number
above plus the per-question detail, because it records, for each ground-truth table:

```json
"MYSCHEMA.FCT_AMOUNT": {
  "final": 12, "vector": null, "keyword": 34, "name_alias": 3, "via_er": false
}
```

A **rank**, not a boolean — that is what makes it possible to say *how far off* a miss
was, and which retriever already had the answer. `null` means absent from that list.

Also send 3–5 of the validator's `-failure.txt` files for the worst cases (the ones with tables in
`unreachable`), since those carry the full stored metadata and document for the missing
table, which is what shows *why* nothing matched it.

> If the question text or table names cannot leave the machine: the analysis only needs
> `totals` and `ranks`. Strip `question`, `gt_tables` and the `ranks` keys down to
> `t1`, `t2`, … before sending — the numbers survive, the content does not.

## 5. The sweep

`--set FIELD=VALUE` overrides one `SearchConfig` field, and repeats for more. Each run
writes one JSON; name them by `--label` and keep every one.

Run the phases **in order** — each one's answer changes what is worth trying next.

Phases 2–5 all take the same shape, so define this once and reuse it. It holds
`--top-n` fixed at whatever Phase 1 settled on, which is what makes the runs comparable:

```bash
run () {                          # run <label> [--set flags…]
  local label=$1; shift
  ENV=uat .venv/bin/python -m app.rag.test.table_retrieval_analyser \
      --top-n 30 --label "$label" --out "./results/$label.json" "$@"
}
```

### Phase 1 — depth (cheap, no rebuild)

```bash
for n in 10 20 30 50; do
  ENV=uat .venv/bin/python -m app.rag.test.table_retrieval_analyser \
    --top-n $n --label "topn-$n" --out ./results/topn-$n.json
done
```

Answers: how deep must the caller read? Where does `recall@k` flatten?

### Phase 2 — retriever ablation (the highest-information runs)

One retriever off per run. Four runs tell you what each is actually worth, which no
amount of weight-fiddling will reveal:

```bash
run vector-only    --set keyword_weight=0
run lexical-only   --set vector_weight=0
run no-name-alias  --set name_alias_weight=0
run no-columns     --set column_weight=0
run no-ngram       --set ngram_weight=0 --set name_ngram_weight=0
```

A retriever whose removal does not move `micro_recall` is not earning its build cost.
The n-gram pair is the one to scrutinise: it is the most expensive to build.

### Phase 3 — weights, around whatever Phase 2 showed matters

Only after ablation, and only on the retrievers ablation proved matter. Remember the
weights are **local to their node in the fusion tree** (`TABLE-SEARCH-CODEBASE.md` §1),
so doubling `column_weight` does not double its influence on the final ranking:

One run per value — these are three separate runs each, not one:

```bash
for w in 0.5 1.0 2.0;   do run "namealias-$w" --set name_alias_weight=$w; done
for w in 0.15 0.3 0.6;  do run "ngram-$w"     --set ngram_weight=$w;      done
for k in 10 60 120;     do run "rrfk-$k"      --set rrf_k=$k;             done
```

`rrf_k` low = trust each retriever's top hit; high = reward agreement between them.

### Phase 4 — ER expansion

```bash
run er-off    --set er_candidates_n=0                      # off entirely
run er-5-25   --set er_anchor_n=5  --set er_related_n=25   # more graph, fewer anchors
run er-20-10  --set er_anchor_n=20 --set er_related_n=10   # the reverse
```

Watch `gt_via_er` against `micro_recall`. ER earns its precision cost only if the
tables it adds are ground truth. Note `er_anchor_n` comes out of the same `top_n`
budget, so these must be compared at a fixed `--top-n`.

### Phase 5 — build-time knobs (each needs an index rebuild)

`bm25_b`, `bm25_k1`, `ngram_range`, `keyword_fields` are baked into the cached matrices.
Changing one changes the cache fingerprint, so the next run refits the whole index —
correct, but slow. Budget for it, and do these last:

```bash
for b in 0.0 0.4 0.75; do run "bm25b-$b" --set bm25_b=$b; done
run no-rules --set keyword_fields=name,alias,domain,description   # i.e. drop `rules`
```

`bm25_b` is length normalisation, `0`–`1`; ours sits low because the documents are short
and uneven. Each of these four runs refits the whole index before it scores anything.

## 6. Reading the results — what each pattern means

| Pattern in `summary.json` | Reading |
|---|---|
| GT ranked well in one retriever, `final` null | Fusion weights. Winnable. Phase 3. |
| GT `null` everywhere, `unreachable` | Not a search problem. The table has no matchable surface — check its description, aliases, `columns` metadata, and the glossary. |
| `via_er: true` on many GT tables | ER is carrying the result. Check it is not also adding noise: compare `micro_recall` with `er_candidates_n=0`. |
| `recall@1` low but `recall@10` high | Ordering problem, not retrieval. Look at `rrf_k` and the weights, not at depth. |
| `deepest_rank` near `top_n` on many questions | `top_n` is the binding constraint; Phase 1. |
| Big `vector` vs `keyword` recall gap | The gap direction says which side to weight up — and whether the embedded document (name/aliases/description, no columns) is carrying its weight. |

## 7. Caveats worth stating in any write-up

- **Sample size.** With a limited question set, ground-truth *tables* are the unit —
  roughly `questions × tables per question`. A one- or two-table difference between
  configs is noise. Prefer a config that wins on several phases over one that wins by a
  hair on the metric being tuned.
- **Overfitting.** The same questions are being used to choose the config and to report
  the result. Numbers from a swept config are not an unbiased estimate of live
  performance. If the set is large enough, hold a third out and only run the winner
  against it at the end.
- **Strict success is harsh.** It requires *every* ground-truth table. `micro_recall`
  and the `recall@k` curve are the more informative comparators between configs; keep
  strict success as the headline, not as the thing being optimised.
- **One variable per run.** `--set` composes, which makes it easy to move two knobs and
  learn nothing about either.
