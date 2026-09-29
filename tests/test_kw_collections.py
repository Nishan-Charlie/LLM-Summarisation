"""Imbalanced collections: the manipulated variable.

These tests pin the properties the study's causal claim rests on. Every
collection has exactly its level's split, dev and test never share an entity,
and the same seed rebuilds the same collections.
"""

import pandas as pd
import pytest

from fairsum.kwsum.collections import (
    LEVELS,
    Collection,
    InsufficientEntitiesError,
    build_collections,
    read_collections,
    split_entities,
    write_collections,
)


def toy_pool(n_entities=40, n_pos=8, n_neg=8) -> pd.DataFrame:
    rows = []
    for e in range(n_entities):
        for polarity, n in (("positive", n_pos), ("negative", n_neg)):
            for i in range(n):
                rows.append({
                    "review_id": f"e{e}-{polarity}-{i}",
                    "dataset": "toy",
                    "entity": f"e{e:02d}",
                    "topic": f"product {e}",
                    "text": f"text {e} {polarity} {i}",
                    "rating": 5.0 if polarity == "positive" else 1.0,
                    "polarity": polarity,
                    "n_words": 40,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def built():
    return build_collections(toy_pool(), "toy", n_test=10, n_dev=3, seed=42)


class TestLevels:
    def test_every_collection_has_its_levels_exact_split(self, built):
        for c in built:
            assert (c.n_pos, c.n_neg) == LEVELS[c.level]

    def test_every_level_and_split_gets_the_requested_count(self, built):
        counts = pd.Series([(c.level, c.split) for c in built]).value_counts()
        for level in LEVELS:
            assert counts[(level, "test")] == 10
            assert counts[(level, "dev")] == 3

    def test_no_review_repeats_within_a_collection(self, built):
        for c in built:
            assert len(set(c.review_ids)) == 8

    def test_an_entity_is_used_at_most_once_per_level_and_split(self, built):
        keys = [(c.level, c.split, c.entity) for c in built]
        assert len(keys) == len(set(keys))

    def test_minority_position_varies(self, built):
        positions = {c.polarities.index("negative") for c in built if c.level == "pos88"}
        assert len(positions) > 1


class TestSplits:
    def test_dev_and_test_share_no_entity(self, built):
        dev = {c.entity for c in built if c.split == "dev"}
        test = {c.entity for c in built if c.split == "test"}
        assert dev and test and not dev & test

    def test_split_entities_is_deterministic_and_sized(self):
        entities = [f"e{i}" for i in range(20)]
        a = split_entities(entities, dev_fraction=0.25, seed=1)
        assert a == split_entities(entities, dev_fraction=0.25, seed=1)
        assert sum(v == "dev" for v in a.values()) == 5


class TestDeterminism:
    def test_same_seed_same_collections(self, built):
        again = build_collections(toy_pool(), "toy", n_test=10, n_dev=3, seed=42)
        assert [c.review_ids for c in again] == [c.review_ids for c in built]

    def test_different_seed_different_collections(self, built):
        other = build_collections(toy_pool(), "toy", n_test=10, n_dev=3, seed=7)
        assert [c.review_ids for c in other] != [c.review_ids for c in built]


class TestInsufficientEntities:
    def test_fails_loudly_naming_the_level_when_too_few_entities_qualify(self):
        pool = toy_pool(n_entities=40, n_pos=8, n_neg=5)  # bal fits, neg75 needs 6
        with pytest.raises(InsufficientEntitiesError, match="neg75"):
            build_collections(pool, "toy", n_test=10, n_dev=3, levels=["bal", "neg75"], seed=42)


class TestCollection:
    def make(self, polarities):
        n = len(polarities)
        return Collection(
            "c", "toy", "test", "x", "e", "t",
            tuple(f"r{i}" for i in range(n)),
            tuple(f"text{i}" for i in range(n)),
            tuple(polarities),
        )

    def test_majority_and_minority(self):
        c = self.make(["positive"] * 7 + ["negative"])
        assert c.majority == "positive" and c.minority == "negative"
        assert c.source_neg_share == pytest.approx(1 / 8)

    def test_balanced_has_no_majority(self):
        c = self.make(["positive", "negative"] * 4)
        assert c.majority is None and c.minority is None

    def test_round_trip_through_jsonl(self, tmp_path, built):
        path = tmp_path / "c.jsonl"
        write_collections(built[:5], path)
        assert read_collections(path) == built[:5]
