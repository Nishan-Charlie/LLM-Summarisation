"""Building the review pool from raw Druglib rows.

v1 filtered to rating-polarity extremes and dropped neutrals, because sampling
conditions were defined over polarity. v2 defines conditions over adverse-event
density instead (`F4 Design Confounds` #3), so the pool keeps every review with
usable text -- there is no reason to discard the middle of the rating scale when
it is not the variable under study.
"""

import pandas as pd
import pytest

from fairsum.data import build_review_pool, druglib_text


def raw_frame(**overrides):
    row = {
        "urlDrugName": "lyrica",
        "rating": 8,
        "effectiveness": "Highly Effective",
        "sideEffects": "Mild Side Effects",
        "condition": "fibromyalgia",
        "benefitsReview": "It helped my pain a great deal " * 5,
        "sideEffectsReview": "Some dizziness in the mornings",
        "commentsReview": "Started at a low dose",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestDruglibText:
    def test_concatenates_the_three_free_text_fields(self):
        text = druglib_text(
            {"benefitsReview": "helped", "sideEffectsReview": "dizzy", "commentsReview": "ok"}
        )
        assert text == "helped dizzy ok"

    def test_skips_missing_fields_without_leaving_gaps(self):
        text = druglib_text(
            {"benefitsReview": "helped", "sideEffectsReview": None, "commentsReview": "ok"}
        )
        assert text == "helped ok"

    def test_returns_empty_when_every_field_is_missing(self):
        assert druglib_text({"benefitsReview": None}) == ""


class TestBuildReviewPool:
    def test_produces_the_columns_the_pipeline_expects(self):
        pool = build_review_pool(raw_frame())
        for column in ("review_id", "product_id", "condition_name", "text", "effectiveness"):
            assert column in pool.columns

    def test_review_ids_are_unique(self):
        frame = pd.concat([raw_frame(), raw_frame(condition="nerve pain")], ignore_index=True)
        pool = build_review_pool(frame)
        assert pool["review_id"].is_unique

    def test_review_ids_are_stable_across_calls(self):
        frame = raw_frame()
        assert list(build_review_pool(frame)["review_id"]) == list(
            build_review_pool(frame)["review_id"]
        )

    def test_keeps_the_per_row_indication_not_the_drugs_first_one(self):
        # F4 #5: a drug treats several conditions and each review names its own.
        frame = pd.concat(
            [raw_frame(condition="fibromyalgia"), raw_frame(condition="nerve pain")],
            ignore_index=True,
        )
        pool = build_review_pool(frame)
        assert set(pool["condition_name"]) == {"fibromyalgia", "nerve pain"}

    def test_drops_reviews_shorter_than_the_word_floor(self):
        frame = raw_frame(
            benefitsReview="short", sideEffectsReview=None, commentsReview=None
        )
        assert len(build_review_pool(frame, min_words=20)) == 0

    def test_drops_reviews_longer_than_the_word_ceiling(self):
        frame = raw_frame(benefitsReview="word " * 500)
        assert len(build_review_pool(frame, max_words=200)) == 0

    def test_drops_rows_with_an_unrecognised_effectiveness_level(self):
        assert len(build_review_pool(raw_frame(effectiveness="Somewhat Okay"))) == 0

    def test_drops_rows_with_no_drug_name(self):
        assert len(build_review_pool(raw_frame(urlDrugName=None))) == 0

    def test_keeps_mid_scale_ratings(self):
        # v1 discarded ratings 5-6 entirely; nothing in v2 depends on polarity.
        assert len(build_review_pool(raw_frame(rating=5))) == 1

    def test_strips_whitespace_from_effectiveness_labels(self):
        assert build_review_pool(raw_frame(effectiveness="  Highly Effective "))[
            "effectiveness"
        ].iloc[0] == "Highly Effective"

    def test_normalises_column_names_with_stray_whitespace(self):
        frame = raw_frame()
        frame.columns = [f" {c} " for c in frame.columns]
        assert len(build_review_pool(frame)) == 1
