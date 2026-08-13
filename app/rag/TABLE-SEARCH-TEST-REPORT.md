# Table search — test report

**Runs:** 2026-08-12 · 
## 1. Objective

Establish how much of the correct table set the search returns for realistic business
questions, and whether the configuration currently in use is the best one available.

## 2. Test conditions

| | |
|---|---|
| Question set | 8 questions with known correct answers, 29 correct tables in total |
| Of which | 1 question is a deliberate negative example (4 tables); the effective set is 7 questions / 25 tables |
| Configurations exercised | 27, each over the whole question set |
| Return depth | 30 tables per question, held fixed except where depth itself was the variable |
| Retrieval under test | dense (embedding) retrieval, four lexical retrievers (description/rules, character n-grams, table name and aliases, column names), rank fusion across all of them, and relationship-graph expansion |
| Catalog | the table catalog as loaded on the test machine at the time of the runs |

**Measure.** For each question, how many of its correct tables appear in the returned
list. Reported as a count over all 29 (or 25) correct tables, together with the rank at
which each was returned and which retriever found it.

## 3. Results

### 3.1 Current configuration

| | all 29 | excluding the negative example (25) |
|---|---|---|
| Correct tables returned | 16 (55%) | 16 (64%) |
| Reachable at all — found by some retriever, at any depth | 24 (83%) | 23 (92%) |
| Found by no retriever at any depth | 5 | **2** |
| Returned within the first 10 results | 4 (14%) | 4 (16%) |

### 3.2 Contribution of each retriever

Measured independently, over all 29 correct tables:

| Retriever | Correct tables it found |
|---|---|
| Lexical (description, rules, name, aliases) | 23 (79%) |
| Table name and aliases alone | 16 (55%) |
| Dense (embeddings) | 6 (21%) |
| Relationship-graph expansion | supplied 12 of the 16 finally returned |

With graph expansion disabled the result falls from 16 to 9 correct tables.

### 3.3 Configuration sweep

No alternative setting improved on the values currently in use. Correct tables returned,
out of 29:

| Setting varied | Values tried | Result |
|---|---|---|
| Return depth | 10 / 20 / **30** / 50 | 7 / 11 / **16** / 14 |
| Rank-fusion constant | 10 / **60** / 120 | 14 / **16** / 11 |
| Name-and-alias weight | 0.5 / **1.0** / 2.0 | 16 / **16** / 8 |
| Character n-gram weight | 0.15 / **0.3** / 0.6 | 15 / **16** / 15 |
| Length normalisation | 0.0 / **0.4** / 0.75 | 15 / **16** / 13 |
| Indexed fields | without business rules | 14 |
| Graph expansion split | off / 5-25 / **10-20** / 20-10 | 9 / 12 / **16** / 10 |
| Lexical retrieval | disabled | 4 |

Two settings are materially worse than the current value: doubling the name-and-alias
weight (16 → 8) and raising the rank-fusion constant to 120 (16 → 11).

### 3.4 Two structural observations

**Ordering, not retrieval, is the larger loss.** 16 correct tables are returned within
30 results but only 4 within the first 10. The information is being retrieved and then
ranked too low.

**Greater depth does not help.** Asking for 50 results returned fewer correct tables
than asking for 30. The returned list reserves part of its budget for graph-connected
tables, so depth and that reservation interact; depth alone is not a lever.

### 3.5 Coverage limit

Excluding the negative example, 2 of 25 correct tables were found by no retriever at any
depth. These are not a tuning problem: they carry no text that any question could match,
and reaching them requires catalog work — descriptions, aliases or column metadata —
rather than configuration.

## 4. Open questions and threats to validity

1. **Sample size.** 29 correct tables in total means one table is worth 3.4 percentage
   points. Differences smaller than about two tables are not meaningful, which covers
   several rows of §3.3. Per-question pass/fail moves in steps of 12.5 points and is not
   a usable comparator at this size.
2. **The dense retriever's result is not yet trusted.** Two runs over the same question
   set on the same day disagree: one shows the dense retriever contributing almost
   nothing, the other shows it holding the correct tables near the top of its own list.
   The cause has not been established, so every conclusion in §3.2 and §3.3 that
   concerns the dense side is provisional.
3. **The column retriever was inactive throughout.** The catalog as loaded carried no
   column metadata, so that retriever never ran. Nothing in this report describes its
   value either way.
4. **The catalog tested is not the current one.** It was built by an earlier loader
   revision, which places column names in the embedded text. The current loader changes
   that. Results describe the catalog as tested, not as it will be built next.
5. **Tuning and reporting used the same questions.** There is no held-out set, so the
   figures are an optimistic estimate of live behaviour.
6. **Precision was not measured.** The test counts whether correct tables are returned,
   not how much unrelated material accompanies them. This matters most for graph
   expansion, which supplies three quarters of the current result.

## 5. Retest plan

1. Rebuild the catalog with the current loader and confirm the column metadata is
   present and the column retriever active.
2. Re-run the baseline and confirm that the two test harnesses agree on it, which
   settles open question 2.
3. Produce the rank of every correct table within each retriever, distinguishing "found
   but ranked too low" from "not indexed at all".
4. Sweep the dense weight and the column weight, which only become meaningful once 1–3
   are settled.
5. Re-run the graph-expansion split. The right balance depends on the quality of the
   text ranking beneath it, which is exactly what is in question.
6. Grow the question set and hold part of it back, so the reported figure is not the one
   the configuration was tuned on.

## 6. Summary

The current configuration is the best of the 27 tested; no setting change on offer
improves it. Of the 25 correct tables in the effective question set, 16 are returned and
23 are reachable, so the available gain from ranking and fusion work is 7 tables — a
larger prize than any configuration change measured here. Two tables need catalog work
rather than search work. Three of the six open questions above must be closed before the
figures for the dense and column retrievers can be quoted at all.
