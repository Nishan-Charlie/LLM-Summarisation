"""Output status: which generations carry scores (FairSum invariant)."""

from fairsum.kwsum.compliance import SCORED_STATUSES, classify_output

GOOD = "The cream soothes dry skin and absorbs quickly. A few users disliked the strong smell."


def test_empty():
    assert classify_output("   ", 0, 256) == "empty"


def test_ok():
    assert classify_output(GOOD, 40, 256) == "ok"


def test_hitting_the_token_cap_is_truncated_but_still_scored():
    assert classify_output(GOOD, 256, 256) == "truncated"
    assert "truncated" in SCORED_STATUSES


def test_counts_only_is_not_a_summary():
    assert classify_output("Out of 8 reviews, 6 are positive and 2 are negative.", 20, 256) == "no_summary"


def test_procedural_only_is_not_a_summary():
    assert classify_output("Let's think step by step. Step 1: Count the positive reviews.", 30, 256) == "no_summary"


def test_refusal():
    assert classify_output("I can't help with summarizing medical reviews.", 12, 256) == "refusal"


def test_an_api_refusal_is_a_refusal_even_with_text():
    assert classify_output(GOOD, 40, 256, stop_reason="refusal") == "refusal"
    assert classify_output(GOOD, 40, 256, stop_reason="content_filter") == "refusal"


def test_api_truncation_comes_from_the_stop_reason_not_the_token_count():
    # Thinking tokens inflate API completion counts, so the count is not the signal.
    assert classify_output(GOOD, 900, 256, stop_reason="end_turn") == "ok"
    assert classify_output(GOOD, 40, 256, stop_reason="max_tokens") == "truncated"
    assert classify_output(GOOD, 40, 256, stop_reason="length") == "truncated"


def test_scored_statuses_are_exactly_ok_and_truncated():
    assert SCORED_STATUSES == {"ok", "truncated"}
