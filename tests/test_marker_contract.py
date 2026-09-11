"""Which prompts require a FINAL SUMMARY: marker, and which do not.

Found by running the Day 2 benchmark: marker_ok_rate was 31-56% on
Llama-3.2-1B, and inspecting raw output showed why. The prompt *ended* with
"FINAL SUMMARY:", so the model continued straight into the summary rather than
echoing the marker. Good summaries were being discarded as parse failures.

The marker exists to separate a reasoning scaffold from the summary. `direct`
and `aspect_aware` produce no scaffold, so the whole completion is the summary
and no marker is needed. `refer`, `ssfp`, `ecvp` and `refer_aspect_aware` do
produce scaffold, so for those a missing marker is a genuine parse failure.

This is a per-prompt contract, not a silent fallback: a prompt that promises a
marker and omits it still fails.
"""

import pytest

from fairsum.instances import Instance
from fairsum.prompts import (
    FINAL_SUMMARY_MARKER,
    PROMPT_BUILDERS,
    PROMPT_SPECS,
    ParseStatus,
    parse_final_summary,
    requires_marker,
)


def make_instance():
    return Instance(
        instance_id="i1",
        product_id="lyrica",
        condition_name=None,
        reviews=[{"review_id": f"r{i}", "text": f"body {i}"} for i in range(8)],
    )


class TestMarkerContract:
    def test_scaffold_free_prompts_do_not_require_a_marker(self):
        assert requires_marker("direct") is False
        assert requires_marker("aspect_aware") is False

    def test_reasoning_prompts_require_a_marker(self):
        for name in ("refer", "ssfp", "ecvp", "refer_aspect_aware"):
            assert requires_marker(name) is True

    def test_every_registered_prompt_has_a_spec(self):
        assert set(PROMPT_SPECS) == set(PROMPT_BUILDERS)

    def test_rejects_an_unknown_method_name(self):
        with pytest.raises(KeyError):
            requires_marker("not_a_method")


class TestPromptText:
    def test_a_scaffold_free_prompt_does_not_mention_the_marker(self):
        # Mentioning it invites the model to echo it for no reason.
        assert FINAL_SUMMARY_MARKER not in PROMPT_BUILDERS["direct"](make_instance())

    @pytest.mark.parametrize("name", ["refer", "ssfp", "ecvp", "refer_aspect_aware"])
    def test_a_reasoning_prompt_asks_for_the_marker_on_its_own_line(self, name):
        assert FINAL_SUMMARY_MARKER in PROMPT_BUILDERS[name](make_instance())

    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_no_prompt_ends_with_the_bare_marker(self, name):
        # Ending on the marker is what caused the model to skip echoing it.
        assert not PROMPT_BUILDERS[name](make_instance()).rstrip().endswith(
            FINAL_SUMMARY_MARKER
        )


class TestParsingUnderTheContract:
    def test_marker_free_prompt_treats_the_whole_completion_as_the_summary(self):
        raw = "Patients mostly improved, though several reported dry skin."
        result = parse_final_summary(raw, requires_marker=False)
        assert result.status is ParseStatus.OK
        assert result.summary == raw

    def test_marker_free_prompt_still_honours_a_marker_when_present(self):
        raw = f"some preamble\n{FINAL_SUMMARY_MARKER}\nthe actual summary"
        result = parse_final_summary(raw, requires_marker=False)
        assert result.summary == "the actual summary"

    def test_marker_required_prompt_still_fails_without_one(self):
        result = parse_final_summary("no marker anywhere", requires_marker=True)
        assert result.status is ParseStatus.NO_MARKER
        assert result.summary is None

    def test_empty_completion_fails_even_when_no_marker_is_required(self):
        result = parse_final_summary("   ", requires_marker=False)
        assert result.status is ParseStatus.EMPTY
        assert result.summary is None

    def test_requires_marker_defaults_to_true(self):
        # The strict reading stays the default; opting out must be explicit.
        assert parse_final_summary("no marker").status is ParseStatus.NO_MARKER

    def test_the_real_llama_failure_case_now_parses(self):
        # Verbatim from the Day 2 diagnostic run, previously discarded.
        raw = (
            "This patient's experience with Avita Tretinoin revealed that the "
            "medication provides noticeable improvements in acne, skin texture, "
            "and reduces scarring, although side effects include dryness, "
            "flaking, and sensitivity."
        )
        assert parse_final_summary(raw, requires_marker=False).status is ParseStatus.OK
