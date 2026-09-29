"""Status of a raw generation: which outputs carry scores.

A failure is never a score (FairSum invariant, vault note F1). An output that
is empty, a refusal, or only counts and reasoning with no summary content gets
a status and no metrics. Hitting the token cap is flagged but still scored,
because the text before the cap is a real summary.
"""

from __future__ import annotations

import re

from fairsum.kwsum.propositions import split_summary

SCORED_STATUSES = frozenset({"ok", "truncated"})
MIN_CONTENT_WORDS = 8
_REFUSAL = re.compile(
    r"\b(?:i can(?:'|no)t|i am unable|i'm unable|as an ai\b|i won't be able)", re.IGNORECASE
)


_API_REFUSALS = frozenset({"refusal", "content_filter"})
_API_CAP_HIT = frozenset({"max_tokens", "length"})


def classify_output(raw: str, completion_tokens: int, max_new_tokens: int,
                    stop_reason: str | None = None) -> str:
    """Status of one generation.

    `stop_reason` is the API provider's own signal and wins when present: API
    completion counts include thinking tokens, so for API models the count says
    nothing about truncation. Local generation passes None and the token count
    decides.
    """
    if stop_reason in _API_REFUSALS:
        return "refusal"
    text = (raw or "").strip()
    if not text:
        return "empty"
    split = split_summary(text)
    content_words = sum(len(p.split()) for p in split.propositions)
    if _REFUSAL.search(text[:300].replace("’", "'")) and content_words < 40:
        return "refusal"
    if content_words < MIN_CONTENT_WORDS:
        return "no_summary"
    hit_cap = (stop_reason in _API_CAP_HIT) if stop_reason is not None else (
        completion_tokens >= max_new_tokens)
    return "truncated" if hit_cap else "ok"
