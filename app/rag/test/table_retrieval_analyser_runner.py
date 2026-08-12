"""
Drives the whole evaluation guide (``TABLE-SEARCH-TEST-PLAN.md``) in one command:
the smoke checks (§2), the baseline (§3), then sweep phases 1–5 (§5), one analyser
run per parameter line, in the order the plan prescribes.

The plan is a list of parameter lines to run one at a time; ~30 of them.
This is that list as data, executed in sequence, plus the two things doing it by hand
does not give you: every run's summary lands under one directory with a predictable
name, and the numbers are collated into a comparison table after each phase instead of
being re-read out of thirty JSON files at the end.

**Each run is a subprocess**, not an in-process call. That is the point, not an
oversight: ``set_search_config`` is process-wide and ``KeywordIndexService.warm()``
short-circuits on the already-built index without looking at the config it is handed —
so a phase-5 run following a phase-4 run in the same process would score the *old*
index while reporting the new ``bm25_b``. A fresh process per run re-derives the cache
fingerprint and refits when it must. The cost is one interpreter start per run, which
is noise next to embedding every question.

Runs are resumable: a run whose summary JSON already exists is skipped, so an
interrupted sweep continues where it stopped (``--force`` re-runs everything).

Alongside the JSON it writes ``report.txt``, the same numbers as a compact
fixed-width sheet: no punctuation inside a value, every rate an integer per mille,
and a check code per row (``--verify`` re-reads a copy of the sheet and names the
rows whose digits no longer add up).

CLI:  python -m app.rag.test.table_retrieval_analyser_runner
      python -m app.rag.test.table_retrieval_analyser_runner --phases baseline,2
      python -m app.rag.test.table_retrieval_analyser_runner --limit 20 --out-dir ./results/trial
      python -m app.rag.test.table_retrieval_analyser_runner --list
      python -m app.rag.test.table_retrieval_analyser_runner --render
      python -m app.rag.test.table_retrieval_analyser_runner --verify report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.logger import get_logger

logger = get_logger(__name__)

ANALYSER = "app.rag.test.table_retrieval_analyser"
VALIDATOR = "app.rag.test.validate_table_retrieval"

# Depths shown in the printed table. The CSV keeps every k the analyser reports;
# these four are enough to see whether the recall@k curve has flattened.
COMPARE_KS = ("1", "5", "10", "30")


# —— the plan as data —————————————————————————————————————————————

@dataclass(frozen=True)
class Run:
    """One line of the plan.

    ``args`` holds only what distinguishes this run; ``--label``/``--out`` are derived
    from ``label``/``summary`` so every run's output is where the collation expects it.
    ``{out}`` in an argument is replaced by the resolved ``--out-dir``.
    """

    label: str
    args: Tuple[str, ...] = ()
    module: str = ANALYSER
    summary: Optional[str] = None       # path under --out-dir; set for analyser runs
    question_flags: bool = True         # accepts --limit / --persona-id
    capture: bool = False               # collect stdout instead of streaming it
    min_lines: int = 0                  # captured-output expectation from §2
    always: bool = False                # ignore the "summary exists" skip


@dataclass(frozen=True)
class Phase:
    key: str
    title: str
    note: str                           # what this phase's numbers answer
    runs: Tuple[Run, ...]
    compare: bool = True                # collate its summaries into a table


PHASES: Tuple[Phase, ...] = (
    Phase(
        key="smoke",
        title="Smoke test (§2)",
        note="Catalog loaded, keyword index built, CSVs found. Nothing below is "
             "meaningful until these three pass.",
        compare=False,
        runs=(
            Run("smoke-search", ("utilised amount",), module="app.rag.table_info_search",
                question_flags=False, capture=True, min_lines=1, always=True),
            # "thousands of lines of vocabulary" — a handful means the index built
            # over an empty or half-loaded catalog.
            Run("smoke-vocab", ("--dump-vocab",), module="app.rag.keyword_index",
                question_flags=False, capture=True, min_lines=1000, always=True),
            Run("smoke-analyser", ("--limit", "5"), summary="smoke/summary.json",
                question_flags=False, always=True),
        ),
    ),
    Phase(
        key="baseline",
        title="Baseline (§3)",
        note="Read micro_recall against ceiling (what tuning can win) and "
             "unreachable_gt (what it cannot) before running any sweep.",
        runs=(
            Run("baseline", ("--top-n", "30"), summary="baseline/summary.json"),
            # The one validator run in the plan: per-question diagnostics for the
            # questions the analyser flags. Writes <q_index>-{success,failure}.txt.
            Run("baseline-reports", ("--top-n", "30", "--out-dir", "{out}/baseline/reports"),
                module=VALIDATOR),
        ),
    ),
    Phase(
        key="1",
        title="Phase 1 — depth",
        note="How deep must the caller read? If @10 ≈ @30, top_n is not the lever.",
        runs=(
            Run("topn-10", ("--top-n", "10"), summary="topn-10.json"),
            Run("topn-20", ("--top-n", "20"), summary="topn-20.json"),
            Run("topn-30", ("--top-n", "30"), summary="topn-30.json"),
            Run("topn-50", ("--top-n", "50"), summary="topn-50.json"),
        ),
    ),
    Phase(
        key="2",
        title="Phase 2 — retriever ablation",
        note="A retriever whose removal does not move micro_recall is not earning its "
             "build cost. Scrutinise the n-gram pair: it is the most expensive to build.",
        runs=(
            Run("vector-only", ("--top-n", "30", "--set", "keyword_weight=0"),
                summary="vector-only.json"),
            Run("lexical-only", ("--top-n", "30", "--set", "vector_weight=0"),
                summary="lexical-only.json"),
            Run("no-name-alias", ("--top-n", "30", "--set", "name_alias_weight=0"),
                summary="no-name-alias.json"),
            Run("no-columns", ("--top-n", "30", "--set", "column_weight=0"),
                summary="no-columns.json"),
            Run("no-ngram", ("--top-n", "30", "--set", "ngram_weight=0",
                             "--set", "name_ngram_weight=0"),
                summary="no-ngram.json"),
        ),
    ),
    Phase(
        key="3",
        title="Phase 3 — weights",
        note="Only the retrievers ablation proved matter are worth reading here. "
             "rrf_k low = trust each retriever's top hit; high = reward agreement.",
        runs=(
            Run("namealias-0.5", ("--top-n", "30", "--set", "name_alias_weight=0.5"),
                summary="namealias-0.5.json"),
            Run("namealias-1.0", ("--top-n", "30", "--set", "name_alias_weight=1.0"),
                summary="namealias-1.0.json"),
            Run("namealias-2.0", ("--top-n", "30", "--set", "name_alias_weight=2.0"),
                summary="namealias-2.0.json"),
            Run("ngram-0.15", ("--top-n", "30", "--set", "ngram_weight=0.15"),
                summary="ngram-0.15.json"),
            Run("ngram-0.3", ("--top-n", "30", "--set", "ngram_weight=0.3"),
                summary="ngram-0.3.json"),
            Run("ngram-0.6", ("--top-n", "30", "--set", "ngram_weight=0.6"),
                summary="ngram-0.6.json"),
            Run("rrfk-10", ("--top-n", "30", "--set", "rrf_k=10"), summary="rrfk-10.json"),
            Run("rrfk-60", ("--top-n", "30", "--set", "rrf_k=60"), summary="rrfk-60.json"),
            Run("rrfk-120", ("--top-n", "30", "--set", "rrf_k=120"), summary="rrfk-120.json"),
        ),
    ),
    Phase(
        key="4",
        title="Phase 4 — ER expansion",
        note="Watch gt_via_er against micro_recall: ER earns its precision cost only if "
             "the tables it adds are ground truth. er_anchor_n comes out of the same "
             "top_n budget, so these are only comparable at a fixed --top-n.",
        runs=(
            Run("er-off", ("--top-n", "30", "--set", "er_candidates_n=0"),
                summary="er-off.json"),
            Run("er-5-25", ("--top-n", "30", "--set", "er_anchor_n=5",
                            "--set", "er_related_n=25"), summary="er-5-25.json"),
            Run("er-20-10", ("--top-n", "30", "--set", "er_anchor_n=20",
                             "--set", "er_related_n=10"), summary="er-20-10.json"),
        ),
    ),
    Phase(
        key="5",
        title="Phase 5 — build-time knobs (each run refits the keyword index)",
        note="These change the cache fingerprint, so each run rebuilds the whole index "
             "before it scores anything. Slowest phase by far; last on purpose.",
        runs=(
            Run("bm25b-0.0", ("--top-n", "30", "--set", "bm25_b=0.0"),
                summary="bm25b-0.0.json"),
            Run("bm25b-0.4", ("--top-n", "30", "--set", "bm25_b=0.4"),
                summary="bm25b-0.4.json"),
            Run("bm25b-0.75", ("--top-n", "30", "--set", "bm25_b=0.75"),
                summary="bm25b-0.75.json"),
            Run("no-rules", ("--top-n", "30", "--set",
                             "keyword_fields=name,alias,domain,description"),
                summary="no-rules.json"),
        ),
    ),
)

# Names that read better than the plan's numbering on a command line.
ALIASES = {"depth": "1", "ablation": "2", "weights": "3", "er": "4", "build": "5"}


def select_phases(spec: str) -> List[Phase]:
    """Resolve a ``--phases`` string to phases, in plan order.

    Order is the plan's, not the caller's: each phase's answer changes what is worth
    reading in the next, and a sweep run out of order is a sweep read out of order.
    """
    if spec.strip().lower() in ("", "all"):
        return list(PHASES)

    by_key = {p.key: p for p in PHASES}
    wanted = set()
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        name = ALIASES.get(name, name)
        if name not in by_key:
            raise SystemExit(
                f"unknown phase {raw.strip()!r}. Valid: "
                f"{', '.join(p.key for p in PHASES)} (aliases: "
                f"{', '.join(f'{a}={k}' for a, k in ALIASES.items())}), or all"
            )
        wanted.add(name)
    return [p for p in PHASES if p.key in wanted]


# —— running one line ——————————————————————————————————————————————

def build_command(run: Run, out_dir: str, args: argparse.Namespace) -> List[str]:
    """The argv the plan would have had you type, with label and out filled in."""
    cmd = [sys.executable, "-m", run.module]
    cmd += [a.replace("{out}", out_dir) for a in run.args]
    if run.summary:
        cmd += ["--label", run.label, "--out", os.path.join(out_dir, run.summary)]
    if run.question_flags:
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.persona_id is not None:
            cmd += ["--persona-id", str(args.persona_id)]
    return cmd


def execute(run: Run, phase: Phase, out_dir: str, args: argparse.Namespace) -> dict:
    """Run one line and return its record for the manifest."""
    summary_path = os.path.join(out_dir, run.summary) if run.summary else None
    cmd = build_command(run, out_dir, args)
    record = {
        "phase": phase.key,
        "label": run.label,
        "command": " ".join(cmd),
        "summary": summary_path,
        "status": "ok",
        "returncode": 0,
        "seconds": 0.0,
    }

    if summary_path and not run.always and not args.force and os.path.exists(summary_path):
        logger.info("[%s] SKIP — %s already exists (--force to re-run).",
                    run.label, summary_path)
        record["status"] = "skipped"
        return record

    logger.info("[%s] %s", run.label, record["command"])
    if args.dry_run:
        record["status"] = "dry-run"
        return record

    if summary_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)) or ".", exist_ok=True)

    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if run.capture else None,
        stderr=subprocess.STDOUT if run.capture else None,
        text=True,
    )
    record["seconds"] = round(time.monotonic() - started, 1)
    record["returncode"] = proc.returncode

    if proc.returncode != 0:
        record["status"] = "failed"
        if run.capture and proc.stdout:
            logger.error("[%s] output tail:\n%s", run.label, proc.stdout[-2000:])
        logger.error("[%s] FAILED — exit %d after %.1fs.", run.label,
                     proc.returncode, record["seconds"])
        return record

    if run.capture:
        n_lines = len([ln for ln in (proc.stdout or "").splitlines() if ln.strip()])
        record["output_lines"] = n_lines
        if n_lines < run.min_lines:
            record["status"] = "failed"
            logger.error(
                "[%s] FAILED — expected at least %d line(s) of output, got %d. "
                "The catalog or the keyword index is not loaded; see §1 of the plan "
                "(python -m app.rag.vectordb_loader).",
                run.label, run.min_lines, n_lines,
            )
            return record
        logger.info("[%s] OK — %d line(s) of output in %.1fs.",
                    run.label, n_lines, record["seconds"])
        return record

    logger.info("[%s] OK — %.1fs.", run.label, record["seconds"])
    return record


# —— collation ————————————————————————————————————————————————————

def load_summary(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def render_table(rows: List[Tuple[str, dict]]) -> str:
    """Fixed-width comparison of the runs that produced a summary."""
    header = (f"{'label':<16}{'success':>9}{'micro':>8}{'ceiling':>9}"
              + "".join(f"{'@' + k:>7}" for k in COMPARE_KS)
              + f"{'via_er':>8}{'unreach':>9}{'questions':>11}")
    lines = [header, "-" * len(header)]
    for label, totals in rows:
        at = totals.get("recall_at", {})
        lines.append(
            f"{label:<16}"
            f"{100 * totals.get('strict_success', 0):>8.1f}%"
            f"{totals.get('micro_recall', 0):>8.3f}"
            f"{totals.get('union_recall', 0):>9.3f}"
            + "".join(f"{at.get(k, float('nan')):>7.3f}" for k in COMPARE_KS)
            + f"{totals.get('gt_via_er', 0):>8}"
            f"{totals.get('unreachable_gt', 0):>9}"
            f"{totals.get('questions', 0):>11}"
        )
    return "\n".join(lines)


def collect_rows(records: List[dict]) -> List[Tuple[str, dict]]:
    """(label, totals) for every record whose summary is on disk, in run order."""
    rows = []
    for rec in records:
        data = load_summary(rec.get("summary"))
        if data:
            rows.append((rec["label"], data.get("totals", {})))
    return rows


def write_comparison_csv(rows: List[Tuple[str, dict]], path: str) -> None:
    """One row per run, every metric the analyser reports — the sweep in one file."""
    if not rows:
        return
    recall_ks = sorted({k for _, t in rows for k in t.get("recall_at", {})}, key=int)
    columns = (["label", "questions", "strict_success", "micro_recall", "union_recall"]
               + [f"recall@{k}" for k in recall_ks]
               + ["vector_recall", "keyword_recall", "name_alias_recall",
                  "gt_via_er", "unreachable_gt"])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for label, t in rows:
            at, per_arm = t.get("recall_at", {}), t.get("retriever_recall", {})
            writer.writerow(
                [label, t.get("questions"), t.get("strict_success"),
                 t.get("micro_recall"), t.get("union_recall")]
                + [at.get(k) for k in recall_ks]
                + [per_arm.get("vector"), per_arm.get("keyword"), per_arm.get("name_alias"),
                   t.get("gt_via_er"), t.get("unreachable_gt")]
            )


def report_worst_questions(out_dir: str, top: int = 5) -> None:
    """Name the validator reports worth sending back (§4).

    The analyser's ``q_index`` and the validator's report file names count the same
    rows in the same order, so the worst questions in the baseline summary name their
    own report files.
    """
    data = load_summary(os.path.join(out_dir, "baseline", "summary.json"))
    if not data:
        return
    worst = sorted(
        (q for q in data.get("questions", []) if q.get("unreachable")),
        key=lambda q: (len(q["unreachable"]), q["n_gt"]), reverse=True,
    )[:top]
    if not worst:
        logger.info("Baseline has no unreachable ground-truth tables — every GT table "
                    "was surfaced by some retriever.")
        return
    logger.info(
        "Worst baseline questions (unreachable ground truth). Send these with the "
        "summaries — they carry the stored metadata for the missing tables:"
    )
    for q in worst:
        logger.info("  %s  (%d of %d GT unreachable: %s)",
                    os.path.join(out_dir, "baseline", "reports",
                                 f"{q['q_index']}-failure.txt"),
                    len(q["unreachable"]), q["n_gt"], ", ".join(q["unreachable"]))


# —— compact text report ——————————————————————————————————————————
#
# JSON is a poor way to move a result set by hand: braces and quotes outnumber the
# digits, one value per line means the numbers never sit on one screen, and a digit
# copied wrong is silently plausible. This block writes the same numbers as a dense
# fixed-width sheet instead — one row per run, every rate an integer per mille, and a
# check code that turns a silent misreading into a loud one.

# Crockford base32 — no I, L, O or U. The decoder folds the glyphs that get confused
# in print (I and L onto 1, O onto 0) back before comparing, so a check code is not
# rejected over a font.
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_B32_FOLD = {"I": "1", "L": "1", "O": "0", "U": "V"}

# Rates worth a column. recall@3 and @20 are in the CSV and the JSON; four depths are
# enough here to see whether the curve has flattened, and every column costs a digit
# that has to be read back correctly.
SWEEP_COLUMNS = ("SUC", "MIC", "CEI", "R1", "R5", "R10", "R30",
                 "VEC", "KEY", "NAL", "ER", "UNR")


def check_code(values: Sequence[int], width: int = 2) -> str:
    """Positional check code over a row of integers, as ``width`` base32 characters.

    Weighted by position on purpose: an unweighted sum survives two columns swapping
    places, which is one of the easier mistakes to make when copying a row out.
    """
    total = sum((i + 1) * (v + 1) for i, v in enumerate(values)) % (32 ** width)
    out = ""
    for _ in range(width):
        total, rem = divmod(total, 32)
        out = _B32[rem] + out
    return out


def fold_code(raw: str) -> str:
    """A check code read back in, with the confusable glyphs folded back."""
    return "".join(_B32_FOLD.get(c, c) for c in raw.strip().upper())


def permille(value) -> int:
    """A 0–1 rate as an integer 0–1000. ``0.710`` reads as ``710``: three digits, no
    decimal point to lose and no leading ``0.`` to mistake for the value."""
    return int(round((value or 0) * 1000))


def sweep_values(totals: dict) -> List[int]:
    """The row of integers behind :data:`SWEEP_COLUMNS`, in that order."""
    at, arms = totals.get("recall_at", {}), totals.get("retriever_recall", {})
    return [
        permille(totals.get("strict_success")),
        permille(totals.get("micro_recall")),
        permille(totals.get("union_recall")),
        permille(at.get("1")), permille(at.get("5")),
        permille(at.get("10")), permille(at.get("30")),
        permille(arms.get("vector")), permille(arms.get("keyword")),
        permille(arms.get("name_alias")),
        int(totals.get("gt_via_er") or 0),
        int(totals.get("unreachable_gt") or 0),
    ]


def rank_token(entry: dict) -> str:
    """One ground-truth table as ``<final rank>`` plus the arms that ranked it.

    ``12vk*`` = twelfth in the final list, found by vector and keyword, admitted by ER.
    ``.n`` = absent from the final list though name/alias had it — a fusion problem.
    ``.`` alone = unreachable: no retriever surfaced it at any depth, which no weight
    will fix. That distinction is the whole reason to carry the misses at all.
    """
    final = entry.get("final")
    token = str(final) if final else "."
    for arm, letter in (("vector", "v"), ("keyword", "k"), ("name_alias", "n")):
        if entry.get(arm):
            token += letter
    return token + ("*" if entry.get("via_er") else "")


def miss_values(question: dict, tokens: List[str]) -> List[int]:
    """Integers behind a miss row: the printed counts, then each token as a number.

    A token contributes its rank and a bitmask of its letters, so a dropped ``v`` or a
    lost ``*`` fails the check the same way a wrong digit does.
    """
    values = [question["q_index"], question["n_gt"], question["n_found_final"],
              question.get("deepest_rank") or 0]
    for token in tokens:
        head = token.rstrip("vkn*") or "."
        flags = sum(bit for letter, bit in (("v", 1), ("k", 2), ("n", 4), ("*", 8))
                    if letter in token[len(head):])
        values += [0 if head == "." else int(head), flags]
    return values


def render_text_report(rows: List[Tuple[str, dict]], baseline: Optional[dict],
                      tokens_per_line: int = 8) -> str:
    """The whole sheet: header, one row per run, then the baseline's misses."""
    lines = [
        f"TABLE SEARCH SWEEP   runs={len(rows)}"
        + (f"  questions={baseline['totals'].get('questions')}" if baseline else "")
        + (f"  gt-tables={sum(q['n_gt'] for q in baseline.get('questions', []))}"
           if baseline else ""),
        "rates are per mille (710 = 0.710)   ER UNR are counts   CK = row check",
        "",
        f"{'##':>2}  {'LABEL':<15}" + "".join(f"{c:>5}" for c in SWEEP_COLUMNS) + "   CK",
    ]

    sheet: List[int] = []
    for index, (label, totals) in enumerate(rows, start=1):
        values = sweep_values(totals)
        sheet += [index] + values
        lines.append(
            f"{index:>2}  {label:<15}"
            + "".join(f"{v:>5}" for v in values)
            + f"   {check_code([index] + values)}"
        )
    lines += ["", f"SHEET CK  {check_code(sheet, width=4)}"]

    if not baseline:
        return "\n".join(lines) + "\n"

    misses = [q for q in baseline.get("questions", []) if not q["success"]]
    lines += [
        "",
        f"BASELINE MISSES  {len(misses)} of {baseline['totals'].get('questions')} "
        f"questions   Q names the report file <Q>-failure.txt",
        "token = <final rank or .> + arms that ranked it (v k n) + * if added by ER",
        "",
        f"{'Q':>3}{'GT':>4}{'FND':>5}{'DEEP':>6}  "
        + "RANKS".ljust(6 * tokens_per_line) + " CK",
    ]
    field = 6 * tokens_per_line          # token area, so CK keeps one column of its own
    for question in misses:
        tokens = [rank_token(question["ranks"][table]) for table in question["gt_tables"]]
        code = check_code(miss_values(question, tokens))
        head = (f"{question['q_index']:>3}{question['n_gt']:>4}"
                f"{question['n_found_final']:>5}"
                f"{question.get('deepest_rank') or '.':>6}  ")
        # Wrapped rather than truncated: a row that runs off the screen loses its
        # tail with nothing left to say that it did.
        chunks = [tokens[i:i + tokens_per_line]
                  for i in range(0, len(tokens), tokens_per_line)] or [[]]
        for n, chunk in enumerate(chunks):
            # The continuation prefix is the same width as the head, so the tokens
            # stay in one column down the whole block.
            prefix = head if n == 0 else f"{question['q_index']:>3}   >".ljust(len(head))
            body = "".join(f"{t:<6}" for t in chunk)
            last = n == len(chunks) - 1
            lines.append((prefix + (body.ljust(field) + f" {code}" if last
                                    else body.rstrip())).rstrip())
    return "\n".join(lines) + "\n"


def write_text_report(records: List[dict], out_dir: str, path: str) -> Optional[str]:
    rows = collect_rows(records)
    if not rows:
        return None
    baseline = load_summary(os.path.join(out_dir, "baseline", "summary.json"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_text_report(rows, baseline))
    return path


def render_only(out_dir: str) -> int:
    """Rebuild ``report.txt`` and the CSV from the summaries already on disk.

    A finished sweep is a directory of JSON, and the sheet is only a view of it — so
    wanting the sheet, or a different layout of it, is never a reason to score
    anything a second time.
    """
    if not os.path.isdir(out_dir):
        logger.error("No such directory: %s", out_dir)
        return 1

    records: List[dict] = []
    seen = set()
    for phase in PHASES:
        for run in phase.runs:
            path = os.path.join(out_dir, run.summary) if run.summary else None
            if path and os.path.exists(path):
                records.append({"label": run.label, "summary": path})
                seen.add(os.path.abspath(path))

    # Summaries this runner did not write — an analyser run made by hand carries its
    # own --label, so it still earns a row, after the ones the phases account for.
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if not name.endswith(".json") or name == "runs.json":
            continue
        if os.path.abspath(path) in seen:
            continue
        data = load_summary(path)
        if data and "totals" in data:
            records.append({"label": data.get("label") or name[:-5], "summary": path})

    if not records:
        logger.error("No summary JSON under %s — nothing to render.", out_dir)
        return 1

    rows = collect_rows(records)
    write_comparison_csv(rows, os.path.join(out_dir, "comparison.csv"))
    report = write_text_report(records, out_dir, os.path.join(out_dir, "report.txt"))
    logger.info("All runs\n%s", render_table(rows))
    logger.info("Rendered %d summary file(s) -> %s", len(records), report)
    return 0


# —— verifying a copy ——————————————————————————————————————————————

def verify_report(path: str) -> int:
    """Re-check a copy of the sheet: recompute every row's code from its own digits.

    Whitespace is not trusted (columns rarely survive being copied), so rows are
    read as token sequences: a sweep row is an index, a label, the twelve column
    values and the code; a miss row is its four counts, ``GT`` tokens — continuation
    lines included — and the code.
    """
    with open(path, encoding="utf-8") as fh:
        raw_lines = fh.read().splitlines()

    n_ok = n_bad = n_unparsed = 0
    in_misses = False
    expected_runs: Optional[int] = None
    sheet: List[int] = []
    sheet_code_read: Optional[str] = None
    pending: List[str] = []          # tokens of a miss row still being collected
    pending_head: List[int] = []

    def report(label: str, values: Sequence[int], read: str) -> None:
        nonlocal n_ok, n_bad
        expected = check_code(values)
        if fold_code(read) == expected:
            n_ok += 1
            logger.info("%-22s OK", label)
        else:
            n_bad += 1
            logger.error("%-22s CHECK FAILED — digits give %s, sheet reads %s "
                         "-> re-enter this row", label, expected, fold_code(read))

    for line in raw_lines:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].upper().startswith("TABLE"):
            # "runs=28" in the header is what catches a row dropped altogether:
            # every remaining row can pass its own check and the sheet still be short.
            for token in tokens:
                if token.lower().startswith("runs="):
                    expected_runs = int(token.split("=")[1])
            continue
        if tokens[0].upper().startswith("BASELINE"):
            in_misses = True
            continue
        if tokens[0].upper() == "SHEET" and len(tokens) >= 3:
            sheet_code_read = tokens[2]
            continue

        if not in_misses:
            # index, label, 12 values, code
            if len(tokens) == 15 and tokens[0].isdigit() and all(t.lstrip("-").isdigit()
                                                                for t in tokens[2:14]):
                index, values = int(tokens[0]), [int(t) for t in tokens[2:14]]
                sheet += [index] + values
                report(f"run {tokens[0]} {tokens[1]}", [index] + values, tokens[14])
            elif tokens[0].isdigit():
                # Shaped like a row but not readable as one — a split label or a lost
                # column. Saying nothing would report a short sheet as a clean one.
                n_unparsed += 1
                logger.error("UNPARSED run row (%d token(s), expected 15): %s",
                             len(tokens), line.strip())
            continue

        if pending or (len(tokens) >= 5 and tokens[0].isdigit() and tokens[1].isdigit()):
            if not pending:
                pending_head = [int(tokens[0]), int(tokens[1]), int(tokens[2]),
                                0 if tokens[3] == "." else int(tokens[3])]
                body = tokens[4:]
            else:
                body = tokens[2:] if tokens[1] == ">" else tokens[1:]
            n_gt = pending_head[1]
            code = body.pop() if len(pending) + len(body) > n_gt else None
            pending += body
            if code is None:
                continue
            values = list(pending_head)
            for token in pending:
                head = token.rstrip("vkn*") or "."
                flags = sum(bit for letter, bit in (("v", 1), ("k", 2), ("n", 4), ("*", 8))
                            if letter in token[len(head):])
                values += [0 if head == "." else int(head), flags]
            report(f"question {pending_head[0]}", values, code)
            pending, pending_head = [], []

    n_runs_read = len(sheet) // 13
    if expected_runs is not None and n_runs_read != expected_runs:
        n_bad += 1
        logger.error("%-22s %d run row(s) read, header says %d — %d row(s) are "
                     "missing.", "row count", n_runs_read, expected_runs,
                     expected_runs - n_runs_read)

    if sheet_code_read is not None:
        expected = check_code(sheet, width=4)
        if fold_code(sheet_code_read) == expected:
            logger.info("%-22s OK (%d run row(s))", "sheet", n_runs_read)
        else:
            n_bad += 1
            logger.error("%-22s CHECK FAILED — rows give %s, sheet reads %s. A whole "
                         "row is missing or misread; check the run count in the header.",
                         "sheet", expected, fold_code(sheet_code_read))

    if pending:
        n_bad += 1
        logger.error("%-22s question %s ended mid-row — its continuation line is "
                     "missing.", "miss row", pending_head[0] if pending_head else "?")

    logger.info("Verified %s — %d row(s) OK, %d bad, %d unparsed.",
                path, n_ok, n_bad, n_unparsed)
    return 1 if (n_bad or n_unparsed) else 0


# —— entry point ——————————————————————————————————————————————————

def list_plan(phases: List[Phase], out_dir: str, args: argparse.Namespace) -> int:
    for phase in phases:
        print(f"\n{phase.title}  [--phases {phase.key}]")
        print(f"  {phase.note}")
        for run in phase.runs:
            print("    " + " ".join(build_command(run, out_dir, args)[2:]))
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the table-search evaluation guide end to end: smoke checks, "
                    "baseline, then sweep phases 1-5, one analyser run per parameter "
                    "line, collating the summaries into a comparison table."
    )
    parser.add_argument("--out-dir", default="./results",
                        help="directory for every summary, report and the comparison "
                             "(default ./results)")
    parser.add_argument("--phases", default="all",
                        help="comma-separated subset to run: "
                             + ", ".join(p.key for p in PHASES)
                             + " (aliases: " + ", ".join(ALIASES) + "), or all")
    parser.add_argument("--limit", type=int, default=0,
                        help="only process the first N questions in every run "
                             "(0 = all). For trialling the sweep itself")
    parser.add_argument("--persona-id", type=int, default=None,
                        help="restrict every run to a persona id")
    parser.add_argument("--force", action="store_true",
                        help="re-run runs whose summary JSON already exists "
                             "(default: skip them, so an interrupted sweep resumes)")
    parser.add_argument("--keep-going", action="store_true",
                        help="carry on after a failed run (default: stop, since a "
                             "broken environment fails every remaining run too)")
    parser.add_argument("--dry-run", action="store_true",
                        help="log the commands without running them")
    parser.add_argument("--list", dest="list_only", action="store_true",
                        help="print the phases and their parameter lines, then exit")
    parser.add_argument("--render", action="store_true",
                        help="rebuild report.txt and comparison.csv from the summary "
                             "JSON already in --out-dir, without running anything")
    parser.add_argument("--verify", metavar="FILE",
                        help="check a copy of report.txt instead of running "
                             "anything: recomputes every row's check code from the "
                             "digits read back and names the rows that disagree")
    args = parser.parse_args()

    if args.verify:
        return verify_report(args.verify)
    if args.render:
        return render_only(os.path.abspath(args.out_dir))

    phases = select_phases(args.phases)
    out_dir = os.path.abspath(args.out_dir)

    if args.list_only:
        return list_plan(phases, out_dir, args)

    os.makedirs(out_dir, exist_ok=True)
    n_runs = sum(len(p.runs) for p in phases)
    logger.info("Plan: %d run(s) across %d phase(s) -> %s",
                n_runs, len(phases), out_dir)
    if args.limit:
        logger.info("--limit %d: every run scores only the first %d question(s); the "
                    "numbers are for trialling the sweep, not for choosing a config.",
                    args.limit, args.limit)

    started = time.monotonic()
    records: List[dict] = []
    aborted = False

    for phase in phases:
        logger.info("=== %s ===", phase.title)
        logger.info("%s", phase.note)
        phase_records = []
        for run in phase.runs:
            record = execute(run, phase, out_dir, args)
            records.append(record)
            phase_records.append(record)
            if record["status"] == "failed" and not args.keep_going:
                logger.error("Stopping after %s. Fix it and re-run — completed runs are "
                             "skipped on the next pass (--keep-going to carry on "
                             "regardless).", run.label)
                aborted = True
                break
        if phase.compare:
            rows = collect_rows(phase_records)
            if rows:
                logger.info("%s\n%s", phase.title, render_table(rows))
        if aborted:
            break

    # Written even after an abort: the completed runs are still worth comparing, and
    # the manifest is what says which line stopped the sweep.
    manifest = os.path.join(out_dir, "runs.json")
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({
            "out_dir":    out_dir,
            "phases":     [p.key for p in phases],
            "limit":      args.limit,
            "persona_id": args.persona_id,
            "aborted":    aborted,
            "runs":       records,
        }, fh, indent=2, ensure_ascii=False)

    all_rows = collect_rows(records)
    comparison = os.path.join(out_dir, "comparison.csv")
    write_comparison_csv(all_rows, comparison)
    if all_rows:
        logger.info("All runs\n%s", render_table(all_rows))

    # The sheet is the copy meant to be read and moved by hand; JSON stays here.
    report = write_text_report(records, out_dir, os.path.join(out_dir, "report.txt"))
    if report:
        logger.info("Text sheet -> %s (check a copy of it with --verify)", report)

    if any(p.key == "baseline" for p in phases):
        report_worst_questions(out_dir)

    counts: Dict[str, int] = {}
    for rec in records:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    logger.info(
        "Done in %.1f min — %s. Summaries -> %s | comparison -> %s | manifest -> %s",
        (time.monotonic() - started) / 60,
        ", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "nothing run",
        out_dir, comparison, manifest,
    )
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
