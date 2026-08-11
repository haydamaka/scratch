"""
Adds tables the relationship graph connects to the best text matches.

Both retrievers score a table on how well its *documented surface* matches the
question. A table that belongs in the answer only because the expected SQL
joins it has no such surface: ``fct_amount`` is a ground-truth table for
questions about tenor and utilisation, yet its name, description and column
list mention neither — it is reached through ``fct_facility`` on
``(facility_id, facility_version)``. No embedding or BM25 tuning recovers that
class of table, because there is nothing to match.

So the ranking is read in two parts. Its top entries are kept unconditionally and
called the **anchors** — those are the tables the question actually describes.
Below them, a much deeper slice is added only where the relationship graph
connects a candidate to one of those anchors.

Ranking the added tables by text score alone would be self-defeating: those
scores are precisely what failed to surface them. Instead each candidate is
scored on four bounded signals (see ``_candidate_score``), of which text rank is
only one. Each is normalised to [0, 1], so a weight is directly the number of
"rank places" it is worth; the three boosts sum to ``er_weight_rank`` on purpose, so
a dimension table ranked last can reach — but never exceed — an unboosted table
ranked first.

Public API:
    ``select_with_related_tables(candidate_ids, n_anchors, n_related, cfg)``
        ``(anchors, related, used_graph)``
"""

from __future__ import annotations

import re
from collections import Counter

from app.core.logger import get_logger

logger = get_logger(__name__)

# This is passed here: graph.get_table_connections(table, max_connections=_ALL_CONNECTIONS)
_ALL_CONNECTIONS = 10_000

# Split a table name into segments on non-alphanumeric characters
_RE_SEGMENT = re.compile(r"[^a-z0-9]+")


def _normalize(table: str) -> str:
    """Graph nodes are stored lowercased and stripped."""
    return (table or "").lower().strip()


def is_dim_table(table: str) -> bool:
    name = _RE_SEGMENT.split(_normalize(table).split(".")[-1])
    return "dim" in name


def _connected_tables(graph, table: str) -> "dict[str, float]":
    """``{neighbour: best edge confidence}`` for every table sharing an edge."""
    info = graph.get_table_connections(table, max_connections=_ALL_CONNECTIONS)
    if not info.get("exists"):
        return {}

    found: "dict[str, float]" = {}
    for edge in info.get("connections") or ():
        # Outgoing edges name the far end 'target_table', incoming ones
        # 'source_table'; direction is irrelevant for joinability.
        other = edge.get("target_table") or edge.get("source_table")
        if not other:
            continue
        key = _normalize(other)
        confidence = float(edge.get("confidence") or 0.0)
        found[key] = max(found.get(key, 0.0), confidence)
    return found


def _candidate_score(
    position: int,
    n_candidates: int,
    table: str,
    connection_count: int,
    confidence: float,
    cfg,
) -> float:
    """Score a candidate on text rank, dimension-ness, connections, confidence.
    """
    rank_term = 1.0 - (position / n_candidates) if n_candidates else 0.0
    dim_term  = 1.0 if is_dim_table(table) else 0.0
    corr_term = min(1.0, max(0, connection_count - 1) / max(1, cfg.er_connections_for_full_score))

    return (
        cfg.er_weight_rank      * rank_term
        + cfg.er_weight_dimension     * dim_term
        + cfg.er_weight_connections * corr_term
        + cfg.er_weight_confidence    * max(0.0, min(1.0, confidence))
    )


def select_with_related_tables(
    candidate_ids: "list[str]",
    n_anchors: int,
    n_related: int,
    cfg,
) -> "tuple[list[str], list[str], bool]":
    """Keep the best text matches, then add tables the graph joins to them.
        candidate_ids: Fused ranking, best first.
        n_anchors: How many top-ranked tables to keep unconditionally — the
            anchors. Everything added below them must connect to one of these.
        n_related: Maximum number of graph-connected tables added below them.
        cfg: Carries the ``er_weight_*`` values and
            ``er_connections_for_full_score``.

    Returns:
        ``(anchors, related, used_graph)``. ``related`` is ordered by
        :func:`_candidate_score`, with text rank breaking ties. ``used_graph`` is
        False when the graph was unusable and the remainder was taken on text
        rank alone — the caller must not then report those tables as related, or
        "the graph contributed N tables" measures the fallback.
    """
    anchors = candidate_ids[:n_anchors]
    remainder = candidate_ids[n_anchors:]

    if not remainder or n_related <= 0:
        return anchors, [], True

    try:
        # Imported here, not at module scope: building the graph reads the
        # relationship source, and this module is otherwise import-light.
        from app.utils.table_relationship_graph_builder_service import get_graph_instance
        graph = get_graph_instance()
    except Exception as exc:
        # Taken on text rank alone, so these must NOT be reported as related —
        # see the `used_graph` flag.
        logger.error(
            "[er_filter] relationship graph unavailable (%s) — taking the top "
            "%d on text rank instead.", exc, n_related,
        )
        return anchors, remainder[:n_related], False

    # How many of the anchors each candidate joins, and how good its best
    # connection to any of them is.
    connection_counts: "Counter[str]" = Counter()
    best_confidence: "dict[str, float]" = {}
    for table in anchors:
        try:
            connections = _connected_tables(graph, table)
        except Exception as exc:
            logger.warning("[er_filter] no connections for '%s': %s", table, exc)
            continue
        for other, confidence in connections.items():
            connection_counts[other] += 1
            best_confidence[other] = max(best_confidence.get(other, 0.0), confidence)

    if not connection_counts:
        logger.info(
            "[er_filter] none of the %d anchors has a graph connection — "
            "taking the top %d on text rank instead.", len(anchors), n_related,
        )
        return anchors, remainder[:n_related], False

    scored: "list[tuple[float, int, str]]" = []
    for position, table_id in enumerate(remainder):
        key = _normalize(table_id)
        if key not in connection_counts:
            continue
        scored.append((
            _candidate_score(position, len(remainder), table_id,
                             connection_counts[key], best_confidence.get(key, 0.0), cfg),
            position,
            table_id,
        ))

    # Descending score; text rank breaks ties, so candidates that score the same
    # keep the order the retrievers produced.
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    related = [table_id for _, _, table_id in scored[:n_related]]

    n_dimensions = sum(1 for t in related if is_dim_table(t))
    logger.info(
        "[er_filter] %d/%d candidates below the anchors are connected to "
        "the top %d; adding %d (%d dimension, %d fact/other).",
        len(scored), len(remainder), len(anchors), len(related),
        n_dimensions, len(related) - n_dimensions,
    )
    if related:
        logger.debug(
            "[er_filter] added: %s",
            ", ".join(
                f"{t}(score={s:.3f},rank={p},connections={connection_counts[_normalize(t)]}"
                f"{',dim' if is_dim_table(t) else ''})"
                for s, p, t in scored[:n_related]
            ),
        )

    return anchors, related, True
