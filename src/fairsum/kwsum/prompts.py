"""Prompts for the nine conditions (spec §4).

REFER's wording is reproduced verbatim from Huang et al. (NewSum 2025, §3,
§4.2 and §5.2), including its "{p} and {n} out of 8 are positive, negative"
phrasing, so our REFER numbers are a replication, not a paraphrase. Keyword
guidance goes before the closing cue "The summary of the reviews is:", which
must stay last for completion-style prompting.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from fairsum.kwsum.collections import Collection

REVIEW_SEPARATOR = " || "
VALUES = "positive, negative"
CLOSING = "The summary of the reviews is:"
COT_PREFIX = "Let's think step by step."


def direct_prompt(c: Collection, guidance: str = "") -> str:
    """REFER's direct prompt, with optional guidance before the closing cue."""
    source = REVIEW_SEPARATOR.join(c.texts).rstrip()
    if not source.endswith((".", "!", "?")):
        source += "."
    parts = [
        f"Reviews about {c.topic}. Each review is separated by ||: {source}",
        "Please write a short text containing the salient information (i.e., a summary).",
    ]
    if guidance:
        parts.append(guidance)
    parts.append(CLOSING)
    return " ".join(parts)


def refer_prefix(n_reviews: int) -> str:
    return (
        f"Let's first determine how many reviews out of {n_reviews} are {VALUES}. "
        "Then, let's generate a balanced summary that accurately reflects the "
        "frequency of these opinions."
    )


def oracle_prefix(c: Collection) -> str:
    return (
        f"{c.n_pos} and {c.n_neg} out of {len(c.texts)} are {VALUES}, "
        "generate a balanced summary reflecting this distribution."
    )


def neutral_counts(c: Collection) -> str:
    """REFER-oracle's true counts, without its instruction to write a balanced summary."""
    return f"{c.n_pos} and {c.n_neg} out of {len(c.texts)} are {VALUES}."


def global_guidance(phrases: Sequence[str]) -> str:
    return f"Consider including the following keywords: {', '.join(phrases)}."


def grouped_guidance(groups: Mapping[str, Sequence[str]]) -> str:
    parts = ["Consider including the following keywords."]
    for group, label in (("positive", "Positive"), ("negative", "Negative")):
        if groups.get(group):
            parts.append(f"{label} reviews mention: {', '.join(groups[group])}.")
    return " ".join(parts)


def _direct(c: Collection, kw: Mapping) -> str:
    return direct_prompt(c)


def _refer(c: Collection, kw: Mapping) -> str:
    return f"{refer_prefix(len(c.texts))} {direct_prompt(c)}"


def _cot_refer(c: Collection, kw: Mapping) -> str:
    return f"{COT_PREFIX} {refer_prefix(len(c.texts))} {direct_prompt(c)}"


def _refer_oracle(c: Collection, kw: Mapping) -> str:
    return f"{oracle_prefix(c)} {direct_prompt(c)}"


def _kw_global(c: Collection, kw: Mapping) -> str:
    return direct_prompt(c, global_guidance(kw["global"]))


def _kw_strat(c: Collection, kw: Mapping) -> str:
    return direct_prompt(c, grouped_guidance(kw["strat"]))


def _kw_strat_neg(c: Collection, kw: Mapping) -> str:
    return direct_prompt(c, grouped_guidance(kw["strat_neg"]))


def _key_refer(c: Collection, kw: Mapping) -> str:
    return f"{refer_prefix(len(c.texts))} {direct_prompt(c, grouped_guidance(kw['strat_neg']))}"


def _key_oracle(c: Collection, kw: Mapping) -> str:
    return f"{oracle_prefix(c)} {direct_prompt(c, grouped_guidance(kw['strat_neg']))}"


CONDITIONS: dict[str, Callable[[Collection, Mapping], str]] = {
    "direct": _direct,
    "refer": _refer,
    "cot_refer": _cot_refer,
    "refer_oracle": _refer_oracle,
    "kw_global": _kw_global,
    "kw_strat": _kw_strat,
    "kw_strat_neg": _kw_strat_neg,
    "key_refer": _key_refer,
    "key_oracle": _key_oracle,
}

def _direct_len(c: Collection, kw: Mapping) -> str:
    if "target_words" not in kw:
        raise KeyError("direct_len needs kw['target_words'], the length-matched word target")
    return direct_prompt(c, f"Write about {kw['target_words']} words.")


def _counts_neutral(c: Collection, kw: Mapping) -> str:
    return f"{neutral_counts(c)} {direct_prompt(c)}"


def _counts_reflect(c: Collection, kw: Mapping) -> str:
    """Amendment 10: refer_oracle with only the word "balanced" removed."""
    return (f"{c.n_pos} and {c.n_neg} out of {len(c.texts)} are {VALUES}, "
            f"generate a summary reflecting this distribution. {direct_prompt(c)}")


#: Amendment 9 controls, kept apart from the nine registered conditions. direct_len matches
#: kw_strat's length on the same collection; counts_neutral is refer_oracle minus "balanced".
CONTROLS: dict[str, Callable[[Collection, Mapping], str]] = {
    "direct_len": _direct_len,
    "counts_neutral": _counts_neutral,
    "counts_reflect": _counts_reflect,
}
ALL_CONDITIONS = {**CONDITIONS, **CONTROLS}
LENGTH_REFERENCE = {"direct_len": "kw_strat"}


def length_target(n_words: float) -> int:
    """A word target rounded to the nearest ten, never below twenty."""
    return max(20, int(round(n_words / 10.0)) * 10)


def length_targets(rows, model: str, condition: str = "kw_strat",
                   scorer_version: str | None = None) -> tuple[dict[str, int], int]:
    """Per-collection word targets from one model's reference summaries, plus the model's
    median target for collections whose reference produced no summary."""
    lengths = {
        r["collection_id"]: r["n_words"] for r in rows
        if r["model"] == model and r["condition"] == condition
        and (scorer_version is None or str(r.get("scorer_version")) == str(scorer_version))
        and r.get("n_words")
    }
    if not lengths:
        raise ValueError(f"no {condition} lengths for {model!r} to match against")
    ordered = sorted(lengths.values())
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {cid: length_target(n) for cid, n in lengths.items()}, length_target(median)


#: The plain keyword set each keyword condition is measured against for uptake.
#: strat_neg shares strat's phrases, so its uptake uses them too (comparability).
KEYWORD_BASE: dict[str, str] = {
    "kw_global": "global",
    "kw_strat": "strat",
    "kw_strat_neg": "strat",
    "key_refer": "strat",
    "key_oracle": "strat",
}


def build_prompt(condition: str, collection: Collection, keywords: Mapping) -> str:
    try:
        builder = ALL_CONDITIONS[condition]
    except KeyError:
        raise KeyError(f"unknown condition {condition!r}; known: {sorted(ALL_CONDITIONS)}") from None
    return builder(collection, keywords)
