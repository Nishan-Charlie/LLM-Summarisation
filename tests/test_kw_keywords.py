"""Keyword extraction.

A fake bag-of-words embedder stands in for the sentence encoder, so these tests
pin the selection rules, not a model's taste.
"""

import zlib

import numpy as np

from fairsum.kwsum.collections import Collection
from fairsum.kwsum.keywords import (
    NEGATION_CUES,
    STOP_WORDS,
    candidate_phrases,
    collection_keywords,
    minority_share,
    mmr_select,
    negated_form,
    stratified_keywords,
)


def fake_embed(texts):
    vecs = np.zeros((len(texts), 256))
    for i, text in enumerate(texts):
        for tok in text.lower().split():
            vecs[i, zlib.crc32(tok.encode()) % 256] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def make_collection(texts, polarities):
    n = len(texts)
    return Collection(
        "c1", "toy", "test", "x", "e", "t",
        tuple(f"r{i}" for i in range(n)), tuple(texts), tuple(polarities),
    )


class TestCandidates:
    def test_no_candidate_starts_or_ends_on_a_stop_word(self):
        for phrase in candidate_phrases("The battery life of this phone is great and it charges fast."):
            toks = phrase.split()
            assert toks[0] not in STOP_WORDS and toks[-1] not in STOP_WORDS

    def test_candidates_are_adjective_noun_phrases(self):
        cands = candidate_phrases(
            "The battery life of this phone is great and fast charging, "
            "but the reviews talked about no relief."
        )
        assert {"battery life", "phone", "fast charging", "relief"} <= set(cands)
        assert "reviews talked" not in cands  # verb fragment
        assert "great" not in cands           # a lone adjective is not a keyphrase

    def test_phrases_do_not_cross_punctuation(self):
        assert "great battery" not in candidate_phrases("It is great. Battery died fast.")

    def test_no_plain_candidate_carries_a_negation_cue(self):
        for phrase in candidate_phrases("I had no side effects and didn't gain weight, not once."):
            assert not set(phrase.split()) & NEGATION_CUES

    def test_curly_apostrophes_are_normalised(self):
        assert not any("’" in p for p in candidate_phrases("I didn’t sleep well at night"))


class TestNegatedForm:
    def test_mostly_negated_phrase_keeps_its_cue(self):
        texts = ["No side effects at all.", "I had no side effects.", "Side effects were mild."]
        assert negated_form("side effects", texts) == "no side effects"

    def test_mostly_plain_phrase_is_unchanged(self):
        texts = ["Side effects were bad.", "Many side effects.", "No side effects for me."]
        assert negated_form("side effects", texts) == "side effects"

    def test_the_span_starts_at_the_nearest_cue(self):
        assert negated_form("side effects", ["I didn't have any side effects."]) == (
            "didn't have any side effects"
        )

    def test_cues_beyond_the_window_do_not_count(self):
        texts = ["Not what I expected but the weight gain was real."]
        assert negated_form("weight gain", texts) == "weight gain"


class TestMMR:
    def test_first_pick_is_the_most_similar_candidate(self):
        cands = ["battery life", "screen", "shipping box"]
        doc = fake_embed(["battery life battery life"])[0]
        assert mmr_select(cands, fake_embed(cands), doc, k=1) == ["battery life"]

    def test_returns_at_most_k_distinct_non_nested_phrases(self):
        cands = ["battery", "battery life", "screen", "price", "sound quality"]
        doc = fake_embed(["battery life screen price"])[0]
        chosen = mmr_select(cands, fake_embed(cands), doc, k=3)
        assert len(chosen) == len(set(chosen)) <= 3
        assert not {"battery", "battery life"} <= set(chosen)

    def test_empty_candidates_give_nothing(self):
        assert mmr_select([], np.zeros((0, 4)), np.zeros(4), k=5) == []


LONG_POS = (
    "Great battery life and bright screen with fast charging and solid build quality overall. "
    "Comfortable grip, lovely design, quick delivery and a helpful manual. "
    "Clear sound, strong signal, cheap price and friendly staff."
)
LONG_NEG = "Terrible customer service and broken charger cable after one week of light use"


class TestStratified:
    def test_equal_quota_per_group(self):
        groups = stratified_keywords(
            [LONG_POS, LONG_POS, LONG_NEG, LONG_NEG],
            ["positive", "positive", "negative", "negative"],
            k=6, embed=fake_embed,
        )
        assert len(groups["positive"]) == 3 and len(groups["negative"]) == 3

    def test_a_short_group_spills_its_quota_to_the_other(self):
        groups = stratified_keywords(
            [LONG_POS, "Terrible smell"], ["positive", "negative"], k=10, embed=fake_embed
        )
        assert len(groups["negative"]) < 5
        assert len(groups["positive"]) + len(groups["negative"]) == 10


class TestCollectionKeywords:
    def test_strat_neg_is_strat_with_negation_attached(self):
        c = make_collection(
            [LONG_POS] * 3 + ["No side effects at all but it did nothing for my pain, total waste of money"],
            ["positive"] * 3 + ["negative"],
        )
        kw = collection_keywords(c, fake_embed, k=6)
        assert kw["collection_id"] == "c1"
        assert len(kw["global"]) <= 6
        for group in ("positive", "negative"):
            assert len(kw["strat_neg"][group]) == len(kw["strat"][group])
            for plain, negated in zip(kw["strat"][group], kw["strat_neg"][group]):
                assert negated.endswith(plain)


class TestMinorityShare:
    def test_counts_phrases_found_in_minority_reviews(self):
        c = make_collection(
            ["battery life is great", "battery life is great", "screen cracked badly"],
            ["positive", "positive", "negative"],
        )
        assert minority_share(["battery life", "screen cracked"], c) == 0.5

    def test_balanced_collection_has_no_minority(self):
        c = make_collection(["a b c", "d e f"], ["positive", "negative"])
        assert minority_share(["a b"], c) is None
