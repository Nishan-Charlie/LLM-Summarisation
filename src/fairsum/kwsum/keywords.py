"""Keyphrase extraction for keyword-guided prompting (spec §4.1).

KeyBERT-style: contiguous 1-3-word candidate phrases are ranked by embedding
similarity to the text they summarise, then chosen by maximal marginal
relevance. Implemented here rather than through the keybert package so the
candidate rules are ours to state.

Three keyword sets per collection, each differing from the last in one thing:
  global     k phrases ranked against all 8 reviews
  strat      k/2 per polarity group, each ranked against its own group only
  strat_neg  the same phrases as strat; a phrase whose source occurrences are
             mostly negated is written as its literal negated span
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from fairsum.kwsum.collections import Collection

Embedder = Callable[[Sequence[str]], np.ndarray]

NEGATION_CUES = frozenset({
    "no", "not", "never", "nothing", "none", "without", "hardly", "cannot", "nor",
    "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't",
    "can't", "couldn't", "wouldn't", "shouldn't", "haven't", "hasn't", "hadn't",
})
_EXTRA_STOP = {
    "i'm", "i've", "i'd", "i'll", "it's", "that's", "they're", "you're", "we're",
    "there's", "he's", "she's", "just", "really", "also", "got", "get", "gets",
    "getting", "im", "ive", "dont", "didnt", "doesnt", "thing", "things", "lot", "bit",
}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | _EXTRA_STOP | NEGATION_CUES

_CHUNK_SPLIT = re.compile(r"[.,;:!?()\[\]\"/\n]+|\s-\s")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize_chunks(text: str) -> list[list[str]]:
    """Lowercased tokens, grouped into chunks that a phrase may not cross."""
    text = text.lower().replace("’", "'")
    return [toks for chunk in _CHUNK_SPLIT.split(text) if (toks := _TOKEN.findall(chunk))]


def _has_letter(token: str) -> bool:
    return any(ch.isalpha() for ch in token)


_NOUN_TAGS = frozenset({"NOUN", "PROPN"})
_PHRASE_TAGS = _NOUN_TAGS | {"ADJ"}
_TAGGER = None


def _tagger():
    """spaCy's small English pipeline, loaded once, tagging only."""
    global _TAGGER
    if _TAGGER is None:
        import spacy

        _TAGGER = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    return _TAGGER


def _is_adj_noun_phrase(tags: Sequence[str]) -> bool:
    """ADJ* (NOUN|PROPN)+ : once a noun appears, only nouns may follow."""
    if tags[-1] not in _NOUN_TAGS:
        return False
    seen_noun = False
    for tag in tags:
        if tag in _NOUN_TAGS:
            seen_noun = True
        elif seen_noun:
            return False
    return True


def candidate_phrases(text: str, max_n: int = 3) -> list[str]:
    """Adjective-noun phrases of up to `max_n` words (PatternRank's ADJ* NOUN+).

    Schopf et al. (2022) show part-of-speech-filtered candidates beat plain
    n-grams for KeyBERT-style ranking. Plain n-grams produced fragments such as
    "reviews talked" in the pilot. A phrase may not start or end on a stop
    word or contain a negation cue: negation is `strat_neg`'s variable and must
    not leak into the plain sets. Returned in first-occurrence order.
    """
    doc = _tagger()(text.replace("’", "'"))
    runs: list[list] = [[]]
    for tok in doc:
        if tok.pos_ in _PHRASE_TAGS:
            runs[-1].append(tok)
        elif runs[-1]:
            runs.append([])
    seen: dict[str, None] = {}
    for run in runs:
        for i in range(len(run)):
            for j in range(i + 1, min(len(run), i + max_n) + 1):
                span = run[i:j]
                if not _is_adj_noun_phrase([t.pos_ for t in span]):
                    continue
                toks = _TOKEN.findall(" ".join(t.text for t in span).lower())
                if not toks or toks[0] in STOP_WORDS or toks[-1] in STOP_WORDS:
                    continue
                if not (_has_letter(toks[0]) and _has_letter(toks[-1])):
                    continue
                if any(t in NEGATION_CUES for t in toks):
                    continue
                seen.setdefault(" ".join(toks), None)
    return list(seen)


def _nested(a: str, b: str) -> bool:
    sa, sb = set(a.split()), set(b.split())
    return sa <= sb or sb <= sa


def mmr_select(
    candidates: Sequence[str],
    cand_emb: np.ndarray,
    doc_emb: np.ndarray,
    k: int,
    diversity: float = 0.5,
) -> list[str]:
    """Maximal marginal relevance, skipping phrases nested in a chosen one."""
    if not candidates or k <= 0:
        return []
    sim_doc = cand_emb @ doc_emb
    sim_cc = cand_emb @ cand_emb.T
    selected = [int(np.argmax(sim_doc))]
    remaining = [i for i in range(len(candidates)) if i != selected[0]]
    while len(selected) < k:
        remaining = [
            i for i in remaining
            if not any(_nested(candidates[i], candidates[j]) for j in selected)
        ]
        if not remaining:
            break
        redundancy = sim_cc[np.ix_(remaining, selected)].max(axis=1)
        scores = (1 - diversity) * sim_doc[remaining] - diversity * redundancy
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


def extract_keywords(
    texts: Sequence[str], k: int, embed: Embedder, diversity: float = 0.5
) -> list[str]:
    doc = " ".join(texts)
    candidates = candidate_phrases(doc)
    if not candidates:
        return []
    emb = embed([doc, *candidates])
    return mmr_select(candidates, emb[1:], emb[0], k, diversity)


def stratified_keywords(
    texts: Sequence[str],
    polarities: Sequence[str],
    k: int,
    embed: Embedder,
    diversity: float = 0.5,
) -> dict[str, list[str]]:
    """k/2 phrases per polarity group. A short group's unused quota spills over.

    Each group is ranked against its own reviews only, so a single minority
    review still gets its own phrases instead of competing with seven.
    """
    groups = {
        g: [t for t, p in zip(texts, polarities) if p == g] for g in ("positive", "negative")
    }
    quota = {"positive": k // 2, "negative": k - k // 2}
    ranked = {
        g: extract_keywords(groups[g], k, embed, diversity) if groups[g] else []
        for g in groups
    }
    take = {g: ranked[g][: quota[g]] for g in groups}
    shortfall = {g: quota[g] - len(take[g]) for g in groups}
    for g, other in (("positive", "negative"), ("negative", "positive")):
        if shortfall[g] > 0:
            spare = ranked[other][len(take[other]) :]
            take[other] = take[other] + spare[: shortfall[g]]
    return take


def negated_form(phrase: str, texts: Sequence[str], window: int = 3) -> str:
    """The phrase, or its most frequent negated source span if mostly negated.

    An occurrence is negated when a cue sits in the `window` tokens before it
    (same chunk). The span starts at the nearest such cue, so the keyword reads
    as the reviews wrote it: "no side effects", "didn't have any side effects".
    """
    target = phrase.split()
    n = len(target)
    negated: Counter[str] = Counter()
    plain = 0
    for text in texts:
        for toks in tokenize_chunks(text):
            for i in range(len(toks) - n + 1):
                if toks[i : i + n] != target:
                    continue
                lo = max(0, i - window)
                cues = [j for j in range(lo, i) if toks[j] in NEGATION_CUES]
                if cues:
                    negated[" ".join(toks[cues[-1] : i + n])] += 1
                else:
                    plain += 1
    if sum(negated.values()) > plain:
        return min(negated.items(), key=lambda kv: (-kv[1], len(kv[0])))[0]
    return phrase


def collection_keywords(
    collection: Collection, embed: Embedder, k: int = 10, diversity: float = 0.5
) -> dict:
    strat = stratified_keywords(collection.texts, collection.polarities, k, embed, diversity)
    by_group = {
        g: [t for t, p in zip(collection.texts, collection.polarities) if p == g]
        for g in ("positive", "negative")
    }
    return {
        "collection_id": collection.collection_id,
        "global": extract_keywords(collection.texts, k, embed, diversity),
        "strat": strat,
        "strat_neg": {g: [negated_form(p, by_group[g]) for p in strat[g]] for g in strat},
    }


def _occurs(phrase: str, text: str) -> bool:
    target = phrase.split()
    n = len(target)
    return any(
        toks[i : i + n] == target
        for toks in tokenize_chunks(text)
        for i in range(len(toks) - n + 1)
    )


def minority_share(phrases: Sequence[str], collection: Collection) -> float | None:
    """Share of phrases found in at least one minority review (None if balanced).

    Compared with the minority's share of reviews, this is the pre-generation
    evidence for H1: do salience-ranked keywords under-sample the minority?
    """
    if collection.minority is None or not phrases:
        return None
    minority_texts = [
        t for t, p in zip(collection.texts, collection.polarities) if p == collection.minority
    ]
    hits = sum(any(_occurs(p, t) for t in minority_texts) for p in phrases)
    return hits / len(phrases)


def sentence_transformer_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None
) -> Embedder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            model.encode(list(texts), normalize_embeddings=True, batch_size=128,
                         show_progress_bar=False)
        )

    return embed
