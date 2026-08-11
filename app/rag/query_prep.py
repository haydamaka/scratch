"""
Broadens a question with the words the catalogue is likely to have used instead.

    expand_lexical_query("utilised amount")
    -> "utilised amount outstanding drawn used available"

Public API:
    ``expand_lexical_query(query)``  the query plus its glossary expansions
"""

from __future__ import annotations

import re

from app.core.logger import get_logger

logger = get_logger(__name__)

# term the user types -> terms the catalogue is likely to use instead.
#
# Every entry below is grounded in an observed retrieval failure; grow it the
# same way rather than from intuition. Expansions must be words that actually
# occur in table descriptions/rules — a synonym nothing was written with adds
# query noise and no recall.
GLOSSARY: "dict[str, tuple[str, ...]]" = {
    # "utilized amount" vs descriptions written as "outstanding direct,
    # contingent, and unused commitment amounts" (fct_risk)
    # and "original, available, and released amounts" (fct_amount).
    "utilized":     ("outstanding", "drawn", "used", "available"),
    "utilised":     ("outstanding", "drawn", "used", "available"),
    "utilization":  ("outstanding", "drawn", "used", "available"),
    "utilisation":  ("outstanding", "drawn", "used", "available"),
    # "non-cancelled" facilities vs availability_status_code / _desc.
    "cancelled":    ("availability", "status", "terminated"),
    "canceled":     ("availability", "status", "terminated"),
    # "utilized amount ratio".
    "ratio":        ("proportion", "percentage"),
    # "tenor ... number of months" vs maturity/term prose.
    "tenor":        ("maturity", "term"),
    # "revolver risk rating" — the catalogue writes "revolving".
    "revolver":     ("revolving",),
}

_WORD = re.compile(r"[A-Za-z]+")


def expand_lexical_query(query: str) -> str:
    """Return ``query`` with glossary expansions appended.

    Appended rather than substituted: the user's own wording may well be what
    the catalogue used, so replacing it can only lose matches. Duplicates are
    harmless — the retrievers tokenise and score terms, and a term repeated in a
    short query does not meaningfully change its weight.
    """
    extra: "list[str]" = []
    seen = set()
    for word in _WORD.findall(query.lower()):
        for synonym in GLOSSARY.get(word, ()):
            if synonym not in seen:
                seen.add(synonym)
                extra.append(synonym)

    if not extra:
        return query

    logger.debug("[query_prep] expanded lexical query with: %s", " ".join(extra))
    return f"{query} {' '.join(extra)}"
