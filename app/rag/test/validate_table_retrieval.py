from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

from app.core.logger import get_logger
from app.rag.chroma_db import bootstrap_standalone
from app.rag.table_info_search import get_table_info_search
from app.utils.get_related_tables import get_top_related_tables
from app.utils.table_relationship_graph_builder_service import get_graph_instance

logger = get_logger(__name__)

# The eval CSVs to read the questions from — written by download_questions.py
# and by the eval runs. Columns are matched by name, so the six-column input
# CSV and the wider processed run CSV both read the same way.
QUESTIONS_CSV = str(Path(__file__).resolve().parent / "data" / "*.csv")


def _norm(name: str) -> str:
    """Normalise a table name for comparison (lowercase, strip quotes/backticks)."""
    return name.strip().strip('"').strip("`").lower()


def split_expected_tables(expected_table: str) -> List[str]:
    """EXPECTED_TABLE holds one or more schema.table entries, comma/newline separated.

    Returns them in their original form/order (deduplicated, original casing kept)
    so each can be replayed verbatim as a vector-search query argument.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for part in re.split(r"[,\n]", expected_table or ""):
        part = part.strip()
        if not part:
            continue
        key = _norm(part)
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out

def format_vector_hits(
    hits: List[dict],
    keyword_score_by_norm: Optional[dict] = None,
) -> List[str]:
    """Render vector-search hit dicts as human-readable, indented text lines."""
    if not hits:
        return ["  (no results)"]
    lines: List[str] = []
    for rank, hit in enumerate(hits, start=1):
        name = hit.get("table_name", "?")
        dist = hit.get("distance")
        kw_score = None
        if keyword_score_by_norm:
            kw_score = keyword_score_by_norm.get(_norm(name))

        metric = f"distance={dist}"
        if kw_score is not None:
            metric += f", keyword_score={kw_score}"
        lines.append(f"  [{rank:>2}] {name}  ({metric})")
        desc = (hit.get("table_description") or "").strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:157] + "..."
        if desc:
            lines.append(f"       {desc}")
    return lines


def format_keyword_hits(
    hits: List[dict],
    vector_distance_by_norm: Optional[dict] = None,
) -> List[str]:
    """Render keyword-search hit dicts as human-readable, indented text lines."""
    if not hits:
        return ["  (no results)"]
    lines: List[str] = []
    for rank, hit in enumerate(hits, start=1):
        name = hit.get("table_name", "?")
        score = hit.get("keyword_score")
        dist = None
        if vector_distance_by_norm:
            dist = vector_distance_by_norm.get(_norm(name))

        metric = f"score={score}"
        if dist is not None:
            metric += f", vector_distance={dist}"
        lines.append(f"  [{rank:>2}] {name}  ({metric})")
        desc = (hit.get("table_description") or "").strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:157] + "..."
        if desc:
            lines.append(f"       {desc}")
    return lines


def format_final_hits(hits: List[dict]) -> List[str]:
    """Render the fused + ER-filtered result set — what the search actually returns.

    Unlike the per-arm views this is the product: RRF over the vector and lexical
    arms, then the relationship-graph filter. ``ER`` marks a table admitted by
    that filter rather than by its own text score.
    """
    if not hits:
        return ["  (no results)"]
    lines: List[str] = []
    for rank, hit in enumerate(hits, start=1):
        name = hit.get("table_name", "?")
        marks = [str(hit.get("match_source") or "?")]
        if hit.get("er_expanded"):
            marks.append("ER")
        lines.append(
            f"  [{rank:>2}] {name}  (distance={hit.get('distance')}, "
            f"keyword_score={hit.get('keyword_score')}, "
            f"fused={hit.get('fused_score')}, {'/'.join(marks)})"
        )
    return lines


# Keep backward-compat alias
def format_hits(hits: List[dict]) -> List[str]:
    return format_vector_hits(hits)

def fetch_questions(limit: int) -> List[Tuple]:
    """Read the question rows out of the QUESTIONS_CSV files.

    Same (QUESTION_ID, QUESTION, GROUND_TRUTH_SQL, EXPECTED_TABLE, CATEGORY)
    tuples the QUESTION table used to yield. A question repeated across files is
    kept once, and a row with no QUESTION_ID (the run CSVs leave it blank) gets a
    ``<file stem>-<row>`` id — it only ever names the diagnostic file.
    """
    # A GROUND_TRUTH_SQL cell is a whole SQL script, far past csv's 128 KB cap.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2 ** 31 - 1)

    rows: List[Tuple] = []
    seen: Set[str] = set()

    paths = sorted(glob.glob(QUESTIONS_CSV))
    if not paths:
        logger.warning("No CSV matched %s", QUESTIONS_CSV)

    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            header = [(h or "").strip().upper() for h in (reader.fieldnames or [])]
            if "QUESTION" not in header or "EXPECTED_TABLE" not in header:
                logger.warning("Skipping %s — no QUESTION/EXPECTED_TABLE column.", path)
                continue

            n_file = 0
            for i, raw in enumerate(reader, start=1):
                row = {(k or "").strip().upper(): (v or "")
                       for k, v in raw.items() if k is not None}
                question = row.get("QUESTION", "").strip()
                if not question:
                    continue
                key = " ".join(question.split()).lower()
                if key in seen:
                    continue
                seen.add(key)
                rows.append((
                    row.get("QUESTION_ID", "").strip() or f"{Path(path).stem}-{i}",
                    question,
                    row.get("GROUND_TRUTH_SQL", ""),
                    row.get("EXPECTED_TABLE", ""),
                    row.get("CATEGORY", ""),
                ))
                n_file += 1
        logger.info("Read %d question(s) from %s", n_file, path)

    if limit and limit > 0:
        rows = rows[:limit]
    return rows

def er_distance_note(table: str, hit_tables: List[str], max_hops: int = 5) -> str:
    """Suffix reporting how many relationship edges separate `table` from `hit_tables`.

    The graph's pathfinder optimises for confidence, so the hop count is the
    length of the best-quality path rather than a bare minimum.
    """
    graph = get_graph_instance()
    paths = [p for p in (graph.find_shortest_path(_norm(h), _norm(table), max_hops=max_hops)
                         for h in hit_tables) if p]
    if not paths:
        return f"  (no path within {max_hops} edges of any vector search result)"
    best = min(paths, key=lambda p: p.path_length)
    return f"  (ER distance {best.path_length}: {' -> '.join(best.tables_in_path)})"

def write_failure_report(
        out_dir: str,
        question_id,
        question: str,
        category: str,
        groundtruth_tables: List[str],
        question_hits: List[dict],
        related_tables: List[str],
        report_top_n: int = 20,
        missing_gt_distance_map: Optional[dict] = None,
        missing_gt_record_map: Optional[dict] = None,
        vector_arm_hits: Optional[List[dict]] = None,
        keyword_arm_hits: Optional[List[dict]] = None,
        name_alias_arm_hits: Optional[List[dict]] = None,
) -> str:
    """Write the ``<question_id>.txt`` diagnostic for one no-intersection failure."""
    lines: List[str] = []
    lines.append("Question:")
    lines.append(question)
    lines.append("")
    lines.append(f"Question ID: {question_id}")
    lines.append(f"Category: {category or '(none)'}")
    lines.append("")

    lines.append("Ground-truth tables:")
    for gt in groundtruth_tables:
        lines.append(f"  - {gt}")
    lines.append("")

    # Build lookup sets from top-N only so report semantics match the displayed lists.
    vec_arm = vector_arm_hits or []
    kw_arm  = keyword_arm_hits or []
    na_arm  = name_alias_arm_hits or []
    vec_arm_top = vec_arm[:report_top_n]
    kw_arm_top = kw_arm[:report_top_n]
    na_arm_top = na_arm[:report_top_n]

    vec_arm_norm  = {_norm(h.get("table_name", "")) for h in vec_arm_top}
    kw_arm_norm   = {_norm(h.get("table_name", "")) for h in kw_arm_top}
    na_arm_norm   = {_norm(h.get("table_name", "")) for h in na_arm_top}
    combined_norm = vec_arm_norm | kw_arm_norm | na_arm_norm

    # Top-N lookup maps for GT present/missing sections.
    vec_dist_by_norm_top: dict = {_norm(h["table_name"]): h.get("distance") for h in vec_arm_top}
    kw_score_by_norm_top: dict = {_norm(h["table_name"]): h.get("keyword_score") for h in kw_arm_top}
    na_score_by_norm_top: dict = {_norm(h["table_name"]): h.get("keyword_score") for h in na_arm_top}

    # Full-arm lookup maps for full result-list sections.
    vec_dist_by_norm_all: dict = {_norm(h["table_name"]): h.get("distance") for h in vec_arm}
    kw_score_by_norm_all: dict = {_norm(h["table_name"]): h.get("keyword_score") for h in kw_arm}
    na_score_by_norm_all: dict = {_norm(h["table_name"]): h.get("keyword_score") for h in na_arm}

    groundtruth_norm = {_norm(gt) for gt in groundtruth_tables}
    final_norm = {_norm(h.get("table_name", "")) for h in question_hits}

    # —— Final result set — the one the verdict is judged on ————
    lines.append("Final search results (fused + ER-filtered — what the search returns):")
    lines.extend(format_final_hits(question_hits))
    lines.append("")

    lines.append("Ground-truth tables present in the final search results:")
    present_final = [gt for gt in groundtruth_tables if _norm(gt) in final_norm]
    if present_final:
        er_by_norm = {
            _norm(h.get("table_name", "")): h.get("er_expanded") for h in question_hits
        }
        for gt in present_final:
            via = "  (via ER expansion)" if er_by_norm.get(_norm(gt)) else ""
            lines.append(f"  - {gt}{via}")
    else:
        lines.append("  (none)")
    lines.append("")

    # —— GT present in vector results ——————————————————————————
    lines.append("Ground-truth tables present in the vector search results:")
    present_vec = [gt for gt in groundtruth_tables if _norm(gt) in vec_arm_norm]
    if present_vec:
        for gt in present_vec:
            dist = vec_dist_by_norm_top.get(_norm(gt))
            score = kw_score_by_norm_top.get(_norm(gt))
            if score is None:
                score = na_score_by_norm_top.get(_norm(gt))
            metrics: List[str] = []
            if dist is not None:
                metrics.append(f"vector distance={dist}")
            if score is not None:
                metrics.append(f"keyword score={score}")
            metric_str = f"  ({', '.join(metrics)})" if metrics else ""
            lines.append(f"  - {gt}{metric_str}")
    else:
        lines.append("  (none)")
    lines.append("")

    # —— GT present in vector+keyword results ——————————————————
    lines.append("Ground-truth tables present in the vector+keyword search results:")
    present = [gt for gt in groundtruth_tables if _norm(gt) in combined_norm]
    if present:
        for gt in present:
            dist = vec_dist_by_norm_top.get(_norm(gt))
            score = kw_score_by_norm_top.get(_norm(gt))
            if score is None:
                score = na_score_by_norm_top.get(_norm(gt))
            metrics: List[str] = []
            if dist is not None:
                metrics.append(f"vector distance={dist}")
            if score is not None:
                metrics.append(f"keyword score={score}")
            metric_str = f"  ({', '.join(metrics)})" if metrics else "  (found)"
            lines.append(f"  - {gt}{metric_str}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Ground-truth tables present in the table-name+alias keyword search results:")
    present_na = [gt for gt in groundtruth_tables if _norm(gt) in na_arm_norm]
    if present_na:
        for gt in present_na:
            dist = vec_dist_by_norm_top.get(_norm(gt))
            score = na_score_by_norm_top.get(_norm(gt))
            metrics: List[str] = []
            if dist is not None:
                metrics.append(f"vector distance={dist}")
            if score is not None:
                metrics.append(f"name+alias keyword score={score}")
            metric_str = f"  ({', '.join(metrics)})" if metrics else ""
            lines.append(f"  - {gt}{metric_str}")
    else:
        lines.append("  (none)")
    lines.append("")

    # —— GT missing from vector+keyword results ————————————————
    lines.append("Ground-truth tables missing from the vector+keyword search results:")
    missing_gt = [gt for gt in groundtruth_tables if _norm(gt) not in combined_norm]
    if missing_gt:
        for gt in missing_gt:
            dist = (missing_gt_distance_map or {}).get(gt)
            if dist is None:
                dist = vec_dist_by_norm_top.get(_norm(gt))
            score = kw_score_by_norm_top.get(_norm(gt))
            na_score = na_score_by_norm_top.get(_norm(gt))
            metrics = [
                f"vector distance={dist}" if dist is not None else "vector distance=unavailable",
                f"keyword score={score}" if score is not None else "keyword score=unavailable",
                f"name+alias keyword score={na_score}" if na_score is not None else "name+alias keyword score=unavailable",
            ]
            lines.append(f"  - {gt}  ({', '.join(metrics)})")

            rec = (missing_gt_record_map or {}).get(gt, {})
            if rec.get("found"):
                resolved_name = rec.get("resolved_table_name") or gt
                meta = rec.get("metadata") or {}
                doc = rec.get("document") or ""
                lines.append(f"    resolved_table_name: {resolved_name}")
                lines.append("    full_metadata:")
                meta_json = json.dumps(meta, indent=2, ensure_ascii=True, sort_keys=True)
                for line in meta_json.splitlines():
                    lines.append(f"      {line}")
                lines.append("    full_document:")
                if doc:
                    for line in doc.splitlines():
                        lines.append(f"      {line}")
                else:
                    lines.append("      (empty)")
            else:
                lines.append("    full_metadata: (not found in table_catalog)")
                lines.append("    full_document: (not found in table_catalog)")
    else:
        lines.append("  (none)")
    lines.append("")

    # —— Vector search results ————————————————————————————————
    lines.append("Vector search results for the question:")
    lines.extend(
        format_vector_hits(
            vec_arm if vec_arm else question_hits,
            keyword_score_by_norm={**na_score_by_norm_all, **kw_score_by_norm_all},
        )
    )
    lines.append("")

    # —— Keyword search results ———————————————————————————————
    lines.append("Keyword search results for the question:")
    lines.extend(
        format_keyword_hits(
            kw_arm,
            vector_distance_by_norm=vec_dist_by_norm_all,
        )
    )
    lines.append("")

    lines.append("Table-name+alias keyword search results for the question:")
    lines.extend(
        format_keyword_hits(
            na_arm,
            vector_distance_by_norm=vec_dist_by_norm_all,
        )
    )
    lines.append("")

    # —— Related tables ———————————————————————————————————————
    lines.append("Related tables (get_top_related_tables around vector hits):")
    lines.extend([f"  - {t}" for t in related_tables] or ["  (none)"])
    lines.append("")

    # —— ER-distance section for still-missing GT —————————————
    hits_norm    = {_norm(h.get("table_name", "")) for h in question_hits}
    related_norm = {_norm(t) for t in related_tables}
    hit_names    = [h.get("table_name", "") for h in question_hits]

    lines.append(
        "Ground-truth tables missing from the vector search results + related tables "
        "(with their edge distance to the nearest vector search result):"
    )
    still_missing = [gt for gt in groundtruth_tables if _norm(gt) not in (hits_norm | related_norm)]
    lines.extend(
        [f"  - {gt}{er_distance_note(gt, hit_names)}" for gt in still_missing] or ["  (none)"]
    )
    lines.append("")

    lines.append("")

    path = os.path.join(out_dir, f"{question_id}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path

def main() -> int:
    bootstrap_standalone()
    parser = argparse.ArgumentParser(
        description="For every question whose vector-search results do NOT intersect "
                    "its ground-truth tables, write a per-question diagnostic file."
    )
    parser.add_argument("--top-n", type=int, default=30,
                        help="tables to request from the hybrid search and to display "
                             "in reports. With ER expansion enabled this budget is "
                             "split into an unconditional head of ER_ANCHOR_N plus up "
                             "to ER_EXPAND_N graph-connected tables (default 30 = "
                             "10 + 20). The pool those 20 are drawn from is ER_POOL_N")
    parser.add_argument("--gt-top-n", type=int, default=None,
                        help="tables to request when a ground-truth table name is "
                             "used as the query (default: same as --top-n)")
    parser.add_argument("--limit", type=int, default=0,
                        help="only process the first N questions (0 = all)")
    parser.add_argument("--persona-id", type=int, default=None,
                        help="restrict the vector search to a persona id")
    parser.add_argument("--out-dir", default="./table_retrieval_results",
                        help="directory for the per-question <question_id>.txt files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    search = get_table_info_search()

    rows = fetch_questions(args.limit)
    logger.info("Loaded %d question row(s) from %s.", len(rows), QUESTIONS_CSV)

    n_ok = n_failed = n_skipped = q_index = 0

    for question_id, question, _ground_truth_sql, expected_table, category in rows:
        q_index+=1
        question = question or ""
        expected_table = expected_table or ""

        groundtruth_tables = split_expected_tables(expected_table)
        if not groundtruth_tables:
            n_skipped += 1
            logger.warning("[q=%s] SKIP — EXPECTED_TABLE empty; no ground truth to check.",
                           question_id)
            continue

        groundtruth_norm = {_norm(t) for t in groundtruth_tables}

        question_hits = search.get_search_hit_info(
            question, top_n=args.top_n, persona_id=args.persona_id
        )
        vector_arm_hits, keyword_arm_hits, name_alias_arm_hits = search.get_retriever_hits_detailed(
            question, top_n=args.top_n, persona_id=args.persona_id
        )
        final_search_results = {_norm(h["table_name"]) for h in question_hits}
        vector_arm_norm = {_norm(h["table_name"]) for h in vector_arm_hits[:args.top_n]}
        keyword_arm_norm = {_norm(h["table_name"]) for h in keyword_arm_hits[:args.top_n]}
        name_alias_arm_norm = {_norm(h["table_name"]) for h in name_alias_arm_hits[:args.top_n]}
        combined_search_results = vector_arm_norm | keyword_arm_norm | name_alias_arm_norm
        related_tables = [
            _norm(t["table"])
            for t in get_top_related_tables(table_names=[h["table_name"] for h in question_hits], limit=0)
        ]
        collected_table_info = set(related_tables).union(final_search_results)

        # The verdict is judged on what the search actually returns — the fused,
        # ER-filtered result set — not on the union of the three raw arms, which
        # is up to 3x larger and exists only as a diagnostic. Judging on the arms
        # would credit the search for tables it never hands to the caller.
        success = final_search_results.issuperset(groundtruth_norm)
        if success:
            n_ok += 1
        else:
            n_failed += 1

        # Fractional recall alongside the all-or-nothing verdict: a question with
        # 4 of 5 ground-truth tables is a failure, but not the same failure as one
        # with 0 of 5, and the superset test cannot tell them apart.
        logger.info(
            "[q=%s] recall: final %d/%d, arm-union %d/%d (%d admitted by ER expansion)",
            question_id,
            len(groundtruth_norm & final_search_results), len(groundtruth_norm),
            len(groundtruth_norm & combined_search_results), len(groundtruth_norm),
            sum(1 for h in question_hits if h.get("er_expanded")),
        )
        missing_gt = [gt for gt in groundtruth_tables if _norm(gt) not in combined_search_results]
        missing_gt_distance_map = search.get_query_distances_for_tables(
            question,
            missing_gt,
            persona_id=args.persona_id,
        )
        missing_gt_record_map = search.get_table_records(
            missing_gt,
            persona_id=args.persona_id,
        )

        file_suffix = "-success" if success else "-failure"
        path = write_failure_report(
            args.out_dir, str(q_index) + file_suffix, question, category,
            groundtruth_tables, question_hits, related_tables,
            report_top_n=args.top_n,
            missing_gt_distance_map=missing_gt_distance_map,
            missing_gt_record_map=missing_gt_record_map,
            vector_arm_hits=vector_arm_hits,
            keyword_arm_hits=keyword_arm_hits,
            name_alias_arm_hits=name_alias_arm_hits,
        )

        if success:
            logger.info(
                "[q=%s] PASS — vector+keyword covers all ground-truth tables. -> %s",
                question_id,
                path,
            )
        else:
            logger.error("[q=%s] FAILURE — vector+keyword is missing ground-truth tables. -> %s",
                         question_id, path)

    logger.info(
        "Done. failures=%d, successes=%d, skipped=%d. Reports -> %s/",
        n_failed, n_ok, n_skipped, args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
