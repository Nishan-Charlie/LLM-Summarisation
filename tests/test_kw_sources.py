"""Review pools: what counts as a usable review.

Every filter a review passes before it can enter a collection is tested here.
A collection builder that silently met a bad review would bias the imbalance
levels it exists to control.
"""

import pytest

from fairsum.kwsum.sources import (
    MAX_WORDS,
    MIN_WORDS,
    amazon_polarity,
    amazon_rows,
    build_pool,
    clean_text,
    drugs_polarity,
    drugs_rows,
)


def words(n: int, stem: str = "w") -> str:
    return " ".join(f"{stem}{i}" for i in range(n))


class TestCleanText:
    def test_unescapes_double_escaped_entities(self):
        assert clean_text("I&amp;#039;ve had no issues") == "I've had no issues"

    def test_drops_html_tags_and_collapses_whitespace(self):
        assert clean_text("Great<br />  value<br/>overall") == "Great value overall"

    def test_strips_one_pair_of_wrapping_quotes(self):
        assert clean_text('"It worked."') == "It worked."

    def test_keeps_inner_quotes(self):
        assert clean_text('He said "fine" to me') == 'He said "fine" to me'


class TestPolarity:
    @pytest.mark.parametrize(
        "stars,expected",
        [(5, "positive"), (4, "positive"), (3, None), (2, "negative"), (1, "negative")],
    )
    def test_amazon_stars(self, stars, expected):
        assert amazon_polarity(stars) == expected

    @pytest.mark.parametrize(
        "rating,expected",
        [(10, "positive"), (7, "positive"), (6, None), (5, None), (4, "negative"), (1, "negative")],
    )
    def test_drugs_rating(self, rating, expected):
        assert drugs_polarity(rating) == expected


class TestAmazonRows:
    def record(self, text, rating=5.0, asin="A1"):
        return {"text": text, "rating": rating, "parent_asin": asin}

    def test_keeps_a_usable_review_with_its_product_title(self):
        rows = list(amazon_rows([self.record(words(40))], {"A1": "Magnesium Tablets"}, "fallback"))
        assert len(rows) == 1
        assert rows[0]["topic"] == "Magnesium Tablets"
        assert rows[0]["polarity"] == "positive"
        assert rows[0]["entity"] == "A1"

    def test_falls_back_when_the_title_is_missing(self):
        rows = list(amazon_rows([self.record(words(40))], {}, "a Health and Personal Care product"))
        assert rows[0]["topic"] == "a Health and Personal Care product"

    def test_drops_three_star_reviews(self):
        assert list(amazon_rows([self.record(words(40), rating=3.0)], {}, "x")) == []

    @pytest.mark.parametrize("n", [MIN_WORDS - 1, MAX_WORDS + 1])
    def test_drops_reviews_outside_the_length_window(self, n):
        assert list(amazon_rows([self.record(words(n))], {}, "x")) == []

    @pytest.mark.parametrize("n", [MIN_WORDS, MAX_WORDS])
    def test_keeps_reviews_at_the_window_edges(self, n):
        assert len(list(amazon_rows([self.record(words(n))], {}, "x"))) == 1

    def test_review_ids_are_unique_by_position(self):
        records = [self.record(words(40, "a")), self.record(words(40, "b"))]
        rows = list(amazon_rows(records, {}, "x"))
        assert rows[0]["review_id"] != rows[1]["review_id"]


class TestDrugsRows:
    def test_cleans_text_and_uses_the_drug_as_entity_and_topic(self):
        rec = {
            "Unnamed: 0": 7,
            "drugName": "Mirtazapine",
            "review": '"' + words(35) + ' I&#039;m fine"',
            "rating": 9.0,
        }
        (row,) = list(drugs_rows([rec]))
        assert row["entity"] == row["topic"] == "Mirtazapine"
        assert row["text"].endswith("I'm fine")
        assert row["review_id"] == "drugs:7"

    def test_drops_neutral_ratings(self):
        rec = {"Unnamed: 0": 1, "drugName": "X", "review": words(40), "rating": 5.0}
        assert list(drugs_rows([rec])) == []


class TestBuildPool:
    def row(self, review_id, entity, text):
        return {
            "review_id": review_id, "dataset": "drugs", "entity": entity, "topic": entity,
            "text": text, "rating": 9.0, "polarity": "positive", "n_words": 40,
        }

    def test_drops_exact_duplicate_texts_across_entities(self):
        text = words(40)
        pool = build_pool([self.row("r1", "Brand", text), self.row("r2", "Generic", text)])
        assert list(pool["review_id"]) == ["r1"]

    def test_empty_input_gives_an_empty_pool_with_columns(self):
        pool = build_pool([])
        assert len(pool) == 0
        assert "polarity" in pool.columns
