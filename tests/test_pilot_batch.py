"""Annotation batch selection.

Discovered by checking the first pilot batch against the pool: `build_pool.py`
took the first 15 shuffled drugs regardless of size, and the pool's median drug
has 3 reviews. Half of the 50 pilot reviews came from drugs that can never form
an 8-review instance, so their labels could not be reused in the bulk set, and
the round-robin filled the rest from the few large drugs it had -- 39 of 50
reviews landed on 5 drugs.

Only drugs that can form an instance are eligible now, and a batch either
spreads evenly across them or fails loudly.
"""

import pandas as pd
import pytest

from fairsum.instances import select_agreement_subset, select_annotation_batch

# Shaped like the real pool: a few large drugs and a long tail of tiny ones.
SIZES = {
    "big1": 20, "big2": 12, "big3": 10, "big4": 9,
    "mid": 7, "tiny1": 1, "tiny2": 2, "tiny3": 1,
}
SPLITS = {
    "big1": "calibration", "big2": "dev", "big3": "calibration", "big4": "test",
    "mid": "dev", "tiny1": "dev", "tiny2": "calibration", "tiny3": "dev",
}
INELIGIBLE = {"mid", "tiny1", "tiny2", "tiny3"}


def pool_frame():
    rows = []
    for drug, n in SIZES.items():
        for j in range(n):
            rows.append(
                {
                    "review_id": f"{drug}-{j}",
                    "product_id": drug,
                    "condition_name": "cond",
                    "text": f"review {j} of {drug}",
                    "split": SPLITS[drug],
                }
            )
    return pd.DataFrame(rows)


def draw(frame=None, **kwargs):
    params = {"batch_size": 9, "n_drugs": 3, "n_reviews": 8, "seed": 0}
    params.update(kwargs)
    return select_annotation_batch(pool_frame() if frame is None else frame, **params)


class TestSelectAnnotationBatch:
    def test_only_draws_from_drugs_that_can_form_an_instance(self):
        # The first pilot's failure: labels on drugs no instance can ever use.
        assert not set(draw()["product_id"]) & INELIGIBLE

    def test_never_draws_from_the_test_split(self):
        assert "big4" not in set(draw()["product_id"])

    def test_spreads_reviews_evenly_across_the_chosen_drugs(self):
        counts = draw(batch_size=9, n_drugs=3)["product_id"].value_counts()
        assert set(counts) == {3}

    def test_an_uneven_remainder_differs_by_at_most_one_review(self):
        counts = draw(batch_size=10, n_drugs=3)["product_id"].value_counts()
        assert counts.max() - counts.min() <= 1

    def test_returns_exactly_the_requested_batch_size(self):
        assert len(draw(batch_size=10)) == 10

    def test_no_review_appears_twice(self):
        ids = draw(batch_size=30)["review_id"]
        assert ids.is_unique

    def test_keeps_the_columns_annotators_need(self):
        columns = set(draw().columns)
        assert {"review_id", "product_id", "condition_name", "text"} <= columns

    def test_is_deterministic_for_a_given_seed(self):
        assert list(draw(seed=5)["review_id"]) == list(draw(seed=5)["review_id"])

    def test_different_seeds_draw_different_reviews(self):
        assert set(draw(seed=1)["review_id"]) != set(draw(seed=2)["review_id"])

    def test_does_not_depend_on_row_order(self):
        shuffled = pool_frame().sample(frac=1.0, random_state=3)
        assert list(draw(frame=shuffled)["review_id"]) == list(draw()["review_id"])

    def test_raises_when_too_few_drugs_are_eligible(self):
        # Three eligible non-test drugs exist; asking for four must not quietly
        # fall back to ineligible ones.
        with pytest.raises(ValueError, match="eligible"):
            draw(n_drugs=4)

    def test_raises_when_the_chosen_drugs_cannot_fill_the_batch(self):
        # big1 + big2 + big3 hold 42 reviews.
        with pytest.raises(ValueError, match="fill"):
            draw(batch_size=43, n_drugs=3)


def batch_frame():
    return pd.DataFrame(
        [
            {"review_id": f"{d}-{j}", "product_id": d, "condition_name": "c", "text": "t"}
            for d in ("a", "b", "c")
            for j in range(4)
        ]
    )


class TestSelectAgreementSubset:
    def test_has_the_requested_size_and_comes_from_the_batch(self):
        subset = select_agreement_subset(batch_frame(), size=6, seed=0)
        assert len(subset) == 6
        assert set(subset["review_id"]) <= set(batch_frame()["review_id"])

    def test_spreads_across_the_batchs_drugs(self):
        counts = select_agreement_subset(batch_frame(), size=6, seed=0)["product_id"].value_counts()
        assert set(counts) == {2}

    def test_is_deterministic_for_a_given_seed(self):
        a = select_agreement_subset(batch_frame(), size=6, seed=9)
        b = select_agreement_subset(batch_frame(), size=6, seed=9)
        assert list(a["review_id"]) == list(b["review_id"])

    def test_raises_when_larger_than_the_batch(self):
        with pytest.raises(ValueError, match="fill"):
            select_agreement_subset(batch_frame(), size=13, seed=0)
