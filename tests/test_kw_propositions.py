"""Splitting summaries into scoreable propositions (spec §5).

The cases under "count statements", "meta lines" and "headings" are taken
from real 3B outputs flagged in the final review: the splitter must remove
the same kind of text in every condition, or conditions are scored on
different subsets of their own summaries.
"""

from fairsum.kwsum.propositions import parse_counts, split_summary


class TestSplitting:
    def test_sentences_become_propositions(self):
        s = split_summary("The cream works well on dry skin. It absorbs quickly.")
        assert s.propositions == ("The cream works well on dry skin", "It absorbs quickly")

    def test_contrast_markers_split_clauses(self):
        s = split_summary("Most users found it effective, but several reported headaches.")
        assert s.propositions == ("Most users found it effective", "several reported headaches")

    def test_bullets_and_markdown_are_stripped(self):
        s = split_summary("**Pros:**\n- Works fast for pain relief\n* Cheap compared to others")
        assert s.propositions == ("Works fast for pain relief", "Cheap compared to others")
        assert s.n_meta == 1

    def test_fragments_under_three_words_are_dropped(self):
        assert split_summary("Great. The pills are easy to swallow.").propositions == (
            "The pills are easy to swallow",
        )

    def test_decimal_numbers_are_not_bullets(self):
        assert split_summary("2.5 mg worked well for my sleep.").propositions == (
            "2.5 mg worked well for my sleep",
        )

    def test_short_bullet_content_is_kept(self):
        s = split_summary("- Great value for money\n- Batteries last a long time")
        assert s.propositions == ("Great value for money", "Batteries last a long time")


class TestCountStatements:
    def test_pure_count_statement_is_parsed_and_removed(self):
        s = split_summary("Out of 8 reviews, 6 are positive and 2 are negative. Users like the taste.")
        assert s.counts == ((6, "positive"), (2, "negative"))
        assert s.n_count_statements == 1
        assert s.propositions == ("Users like the taste",)

    def test_of_the_n_phrasing(self):
        assert parse_counts("6 of the 8 reviews are positive") == [(6, "positive")]

    def test_word_numbers(self):
        assert parse_counts("Seven reviews were positive while one was negative") == [
            (7, "positive"), (1, "negative"),
        ]

    def test_count_without_polarity_is_removed_but_not_parsed(self):
        s = split_summary("Six out of 8 reviewers were satisfied. It smells nice though.")
        assert s.counts == ()
        assert s.n_count_statements == 1

    def test_attributive_counts_are_kept_with_a_generic_subject(self):
        s = split_summary(
            "Two reviewers reported positive outcomes, stating that Pylera effectively "
            "eradicated the bacteria. One reviewer had a negative experience with the taste. "
            "Most reviewers (6 out of 8) found it effective for pain relief."
        )
        assert s.propositions == (
            "Some reviewers reported positive outcomes, stating that Pylera effectively "
            "eradicated the bacteria",
            "Some reviewers had a negative experience with the taste",
            "Most reviewers found it effective for pain relief",
        )
        assert s.counts == () and s.n_count_statements == 0

    def test_an_adverb_before_the_polarity_still_counts(self):
        s = split_summary("4 were overwhelmingly negative, citing issues such as poor quality and shedding.")
        assert s.propositions == (
            "Some were overwhelmingly negative, citing issues such as poor quality and shedding",)

    def test_label_counts_are_count_statements(self):
        s = split_summary("Positive reviews: 4\nNegative Reviews: 4/8\nThe cream works well.")
        assert s.counts == ((4, "positive"), (4, "negative"))
        assert s.propositions == ("The cream works well",)

    def test_counts_never_cross_a_line_break(self):
        assert split_summary("Positive reviews: 6\nNegative reviews: 2").counts == (
            (6, "positive"), (2, "negative"))

    def test_instruction_echoes_are_meta_not_counts(self):
        s = split_summary(
            "Let me determine how many reviews out of 8 are positive, negative. "
            "6 and 2 out of 8 are positive, negative, generate a balanced summary. "
            "The pills work well for sleep."
        )
        assert s.counts == ()
        assert s.propositions == ("The pills work well for sleep",)

    def test_numbers_in_product_titles_are_not_counts(self):
        s = split_summary("The Magnesium 32 Ounce has received overwhelmingly negative feedback from users.")
        assert s.counts == ()
        assert s.propositions == (
            "The Magnesium 32 Ounce has received overwhelmingly negative feedback from users",)


class TestMetaLines:
    def test_procedural_lines_are_removed_and_prefixes_stripped(self):
        s = split_summary(
            "Let's think step by step. Step 1: Count the reviews. "
            "Here is a summary: The product is sturdy and well made."
        )
        assert s.propositions == ("The product is sturdy and well made",)
        assert s.n_meta == 3

    def test_per_review_reasoning_is_meta(self):
        s = split_summary(
            "Review 1: Positive. Review 2: Negative. 1. Positive - works well\n"
            "The product helps most users sleep."
        )
        assert s.propositions == ("The product helps most users sleep",)

    def test_only_text_after_the_last_summary_header_is_scored(self):
        s = split_summary(
            "Step 1: count. The first review praises the taste of the syrup.\n\n"
            "Final Summary:\nThe drug works for most users but causes nausea for some."
        )
        assert s.propositions == ("The drug works for most users", "causes nausea for some")

    def test_polarity_label_prefix_is_stripped(self):
        s = split_summary("Positive reviews: The product works well for back pain.")
        assert s.propositions == ("The product works well for back pain",)


class TestHeadings:
    def test_title_case_heading_is_meta(self):
        s = split_summary("Summary of Buspar Reviews\nMost users found it calming.")
        assert s.propositions == ("Most users found it calming",)

    def test_heading_without_the_word_summary_is_meta(self):
        s = split_summary("Overall Experience\nThe device is easy to use.")
        assert s.propositions == ("The device is easy to use",)
        assert s.n_meta == 1


class TestTruncation:
    def test_an_unterminated_tail_is_dropped_when_asked(self):
        text = "The cream works well. It also smells"
        assert split_summary(text, drop_unterminated_tail=True).propositions == ("The cream works well",)
        assert split_summary(text).propositions == ("The cream works well", "It also smells")
