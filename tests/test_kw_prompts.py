"""Prompt builders: REFER verbatim, keyword guidance in the right place."""

import pytest

from fairsum.kwsum.collections import Collection
from fairsum.kwsum.prompts import CLOSING, CONDITIONS, KEYWORD_BASE, build_prompt

REFER_SENTENCE = (
    "Let's first determine how many reviews out of 8 are positive, negative. "
    "Then, let's generate a balanced summary that accurately reflects the "
    "frequency of these opinions."
)

KW = {
    "global": ["sleep", "weight gain"],
    "strat": {"positive": ["better sleep"], "negative": ["side effects"]},
    "strat_neg": {"positive": ["better sleep"], "negative": ["no side effects"]},
}


def collection():
    return Collection(
        "c1", "drugs", "test", "pos88", "Mirtazapine", "Mirtazapine",
        tuple(f"r{i}" for i in range(8)),
        tuple(f"Review number {i}." for i in range(8)),
        ("positive",) * 7 + ("negative",),
    )


class TestDirect:
    def test_uses_refers_direct_wording_and_ends_on_the_closing_cue(self):
        p = build_prompt("direct", collection(), KW)
        assert p.startswith(
            "Reviews about Mirtazapine. Each review is separated by ||: "
            "Review number 0. || Review number 1."
        )
        assert "Please write a short text containing the salient information (i.e., a summary)." in p
        assert p.endswith(CLOSING)

    def test_no_double_period_after_the_source(self):
        assert ". ." not in build_prompt("direct", collection(), KW)


class TestRefer:
    def test_refer_prefixes_the_verbatim_frequency_instruction(self):
        assert build_prompt("refer", collection(), KW).startswith(REFER_SENTENCE + " Reviews about")

    def test_cot_refer_adds_step_by_step_first(self):
        assert build_prompt("cot_refer", collection(), KW).startswith(
            "Let's think step by step. " + REFER_SENTENCE
        )

    def test_oracle_states_the_true_counts(self):
        assert build_prompt("refer_oracle", collection(), KW).startswith(
            "7 and 1 out of 8 are positive, negative, generate a balanced summary "
            "reflecting this distribution."
        )


class TestKeywords:
    def test_global_keywords_sit_just_before_the_closing_cue(self):
        p = build_prompt("kw_global", collection(), KW)
        assert p.endswith("Consider including the following keywords: sleep, weight gain. " + CLOSING)

    def test_stratified_keywords_are_grouped_by_polarity(self):
        p = build_prompt("kw_strat", collection(), KW)
        assert "Positive reviews mention: better sleep." in p
        assert "Negative reviews mention: side effects." in p

    def test_negation_variant_uses_the_negated_forms(self):
        assert "Negative reviews mention: no side effects." in build_prompt("kw_strat_neg", collection(), KW)

    def test_an_empty_group_is_omitted(self):
        kw = {**KW, "strat": {"positive": ["better sleep"], "negative": []}}
        assert "Negative reviews mention" not in build_prompt("kw_strat", collection(), kw)


class TestCombined:
    def test_key_refer_is_refer_plus_negation_keywords(self):
        p = build_prompt("key_refer", collection(), KW)
        assert p.startswith(REFER_SENTENCE)
        assert "no side effects" in p and p.endswith(CLOSING)

    def test_key_oracle_is_oracle_plus_negation_keywords(self):
        p = build_prompt("key_oracle", collection(), KW)
        assert p.startswith("7 and 1 out of 8") and "no side effects" in p


class TestRegistry:
    def test_all_nine_conditions_are_registered(self):
        assert set(CONDITIONS) == {
            "direct", "refer", "cot_refer", "refer_oracle",
            "kw_global", "kw_strat", "kw_strat_neg", "key_refer", "key_oracle",
        }

    def test_keyword_base_covers_exactly_the_keyword_conditions(self):
        assert set(KEYWORD_BASE) == {"kw_global", "kw_strat", "kw_strat_neg", "key_refer", "key_oracle"}

    def test_unknown_condition_names_the_known_ones(self):
        with pytest.raises(KeyError, match="kw_global"):
            build_prompt("nope", collection(), KW)
