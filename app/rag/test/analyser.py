"""
Measures table retrieval over the eval question set and writes one machine-readable
summary per run, for comparing search configurations.

Where ``validate_table_retrieval`` answers "did this question pass, and why not" with
a per-question text report, this answers "which configuration is better" with numbers.
It reads the same CSVs through the same loader and writes no text reports.

The unit of measurement is a **rank**, not a hit/miss: for every ground-truth table it
records where that table landed in each retriever and in the final list. A rank says how
far off a miss was and which retriever already had the answer — which is what separates
"the weights are wrong" from "nothing can match this table".

CLI:  python -m app.rag.test.analyser --label baseline
      python -m app.rag.test.analyser --set keyword_weight=0 --label vector-only
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields as dataclass_fields
from typing import List, Optional, Tuple

from app.core.logger import get_logger
from app.rag.chroma_db import bootstrap_standalone
from app.rag.search import SearchConfig, get_search_config, set_search_config
from app.rag.table_info_search import get_table_info_search
from app.rag.test.validate_table_retrieval import (
    _norm,
    fetch_questions,
    questions_glob,
    split_expected_tables,
)

logger = get_logger(__name__)

# Depths recall is reported at. The useful question is not "does it work" but
# "how deep must the caller read", so keep a spread.
RECALL_KS = (1, 3, 5, 10, 20, 30)


# —— config overrides —————————————————————————————————————————————

def parse_overrides(assignments: List[str]) -> dict:
    """Turn ``--set name=value`` strings into ``SearchConfig`` keyword arguments.

    Types come from the dataclass defaults, so a typo in a field name fails here with
    the list of valid names rather than silently doing nothing for a whole run.
    """
    defaults = {f.name: f.default for f in dataclass_fields(SearchConfig)}
    out: dict = {}
    for item in assignments or []:
        if "=" not in item:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        name, _, raw = item.partition("=")
        name, raw = name.strip(), raw.strip()
        if name not in defaults:
            raise SystemExit(
                f"unknown config field {name!r}. Valid: {', '.join(sorted(defaults))}"
            )
        current = defaults[name]
        if isinstance(current, bool):
            out[name] = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            out[name] = int(raw)
        elif isinstance(current, float):
            out[name] = float(raw)
        elif isinstance(current, tuple):
            out[name] = tuple(int(x) for x in raw.split(","))
        elif isinstance(current, frozenset):
            out[name] = frozenset(x.strip() for x in raw.split(",") if x.strip())
        else:
            out[name] = raw
    return out


def config_as_dict(cfg: SearchConfig) -> dict:
    """JSON-safe view of a config, recorded in the summary header.

    Without this a summary cannot be attributed to the settings that produced it, which
    is the whole point of running more than one.
    """
    out = {}
    for f in dataclass_fields(cfg):
        value = getattr(cfg, f.name)
        if isinstance(value, frozenset):
            out[f.name] = sorted(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


# —— per-question measurement ——————————————————————————————————————

def rank_of(name_norm: str, hits: List[dict]) -> Optional[int]:
    """1-based rank of a table in a hit list, or None when absent."""
    for rank, hit in enumerate(hits, start=1):
        if _norm(hit.get("table_name", "")) == name_norm:
            return rank
    return None


def measure_question(
    question_id,
    q_index: int,
    question: str,
    category: str,
    groundtruth_tables: List[str],
    final_hits: List[dict],
    vector_hits: List[dict],
    keyword_hits: List[dict],
) -> dict:
    """One record: where each ground-truth table landed in each retriever."""

    per_table = {}
    for gt in groundtruth_tables:
        key = _norm(gt)
        final_rank = rank_of(key, final_hits)
        per_table[gt] = {
            "final":      final_rank,
            "vector":     rank_of(key, vector_hits),
            "keyword":    rank_of(key, keyword_hits),
            # True when the graph put it there, i.e. no retriever ranked it high
            # enough on its own.
        }

    found_final = [t for t, r in per_table.items() if r["final"] is not None]
    # Tables no retriever surfaced at any depth. Weight tuning cannot reach these —
    # they need loader or graph work instead.
    unreachable = [
        t for t, r in per_table.items()
        if r["final"] is None and r["vector"] is None
        and r["keyword"] is None
    ]

    return {
        "q_index":       q_index,
        "question_id":   str(question_id),
        "question":      question,
        "category":      category or "",
        "n_gt":          len(groundtruth_tables),
        "gt_tables":     list(groundtruth_tables),
        "success":       len(found_final) == len(groundtruth_tables),
        "n_found_final": len(found_final),
        # Deepest rank the caller must read to get every GT table it can get.
        "deepest_rank":  max((per_table[t]["final"] for t in found_final), default=None),
        "unreachable":   unreachable,
        "ranks":         per_table,
    }


def aggregate(records: List[dict], recall_ks: Tuple[int, ...] = RECALL_KS) -> dict:
    """Roll per-question records into the numbers a configuration is chosen on."""
    n_gt_total = sum(r["n_gt"] for r in records) or 1
    all_ranks = [entry for r in records for entry in r["ranks"].values()]

    def covered_within(k: int) -> int:
        return sum(1 for e in all_ranks if e["final"] is not None and e["final"] <= k)

    def retriever_recall(arm: str) -> float:
        return sum(1 for e in all_ranks if e[arm] is not None) / n_gt_total

    union_found = sum(
        1 for e in all_ranks
        if any(e[a] is not None for a in ("vector", "keyword"))
    )

    return {
        "questions":        len(records),
        "strict_success":   sum(1 for r in records if r["success"]) / (len(records) or 1),
        "micro_recall":     sum(r["n_found_final"] for r in records) / n_gt_total,
        "recall_at":        {str(k): covered_within(k) / n_gt_total for k in recall_ks},
        "retriever_recall": {a: retriever_recall(a) for a in ("vector", "keyword")},
        # Ceiling: what fusion could reach if the weights were perfect. The gap between
        # this and micro_recall is what tuning can win; the gap from 1.0 to this is what
        # it cannot.
        "union_recall":     union_found / n_gt_total,
        "unreachable_gt":   sum(len(r["unreachable"]) for r in records),
    }


def main() -> int:
    bootstrap_standalone()
    parser = argparse.ArgumentParser(
        description="Score table retrieval over the eval question set and write a "
                    "machine-readable summary, for comparing search configurations."
    )
    parser.add_argument("--top-n", type=int, default=30,
                        help="tables to request from the search (default 30)")
    parser.add_argument("--limit", type=int, default=0,
                        help="only process the first N questions (0 = all)")
    parser.add_argument("--persona-id", type=int, default=None,
                        help="restrict the search to a persona id")
    parser.add_argument("--set", dest="overrides", action="append", metavar="FIELD=VALUE",
                        help="override a SearchConfig field for this run, repeatable "
                             "(e.g. --set keyword_weight=2.0 --set rrf_k=10)")
    parser.add_argument("--questions-dir", default="",
                        help="directory of question CSVs; every *.csv in it is read "
                             "(default app/rag/test/data, or $QUESTIONS_DIR)")
    parser.add_argument("--label", default="",
                        help="name for this run, recorded in the summary")
    parser.add_argument("--out", default="./table_retrieval_results/summary.json",
                        help="path for the summary JSON")
    args = parser.parse_args()

    # Set process-wide, not per call: `search(**overrides)` reaches only the fused
    # path, while the diagnostic retriever views read the singleton — a sweep that set
    # only one of them would compare two different configurations at once.
    cfg = get_search_config().with_overrides(**parse_overrides(args.overrides))
    set_search_config(cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    search = get_table_info_search()

    rows = fetch_questions(args.limit, args.questions_dir or None)
    logger.info("Loaded %d question row(s) from %s.", len(rows),
                questions_glob(args.questions_dir or None))

    records: List[dict] = []
    n_skipped = q_index = 0

    for question_id, question, _ground_truth_sql, expected_table, category in rows:
        q_index += 1
        groundtruth_tables = split_expected_tables(expected_table or "")
        if not groundtruth_tables:
            n_skipped += 1
            logger.warning("[q=%s] SKIP — EXPECTED_TABLE empty; no ground truth to check.",
                           question_id)
            continue

        question = question or ""
        final_hits = search.get_search_hit_info(
            question, top_n=args.top_n, persona_id=args.persona_id
        )
        vector_hits, keyword_hits = search.get_retriever_hits_detailed(
            question, top_n=args.top_n, persona_id=args.persona_id
        )

        record = measure_question(
            question_id, q_index, question, category, groundtruth_tables,
            final_hits, vector_hits, keyword_hits,
        )
        records.append(record)

        logger.info(
            "[q=%s] %s — final %d/%d (%d via ER, deepest rank %s)",
            question_id, "PASS" if record["success"] else "MISS",
            record["n_found_final"], record["n_gt"],
            record["n_via_er"], record["deepest_rank"],
        )

    totals = aggregate(records)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "label":      args.label,
            "top_n":      args.top_n,
            "persona_id": args.persona_id,
            "config":     config_as_dict(cfg),
            "totals":     totals,
            "questions":  records,
        }, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Done. questions=%d, skipped=%d | strict success %.1f%% | micro recall %.3f "
        "| ceiling (union of retrievers) %.3f | %d GT via ER | %d GT unreachable by any "
        "retriever. Summary -> %s",
        totals["questions"], n_skipped,
        100 * totals["strict_success"], totals["micro_recall"], totals["union_recall"],
        totals["gt_via_er"], totals["unreachable_gt"], args.out,
    )
    logger.info(
        "recall@k (final list): %s",
        "  ".join(f"@{k}={totals['recall_at'][str(k)]:.3f}" for k in RECALL_KS),
    )
    logger.info(
        "per-retriever recall @top_n: vector=%.3f keyword=%.3f",
        totals["retriever_recall"]["vector"],
        totals["retriever_recall"]["keyword"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
