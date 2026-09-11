"""Prompt construction and output parsing.

v1 scored whatever the model emitted, so prompts that instruct the model to
enumerate counts and run checklists first (SSFP, ECVP) spent their 256-token
budget on scaffolding and got truncated mid-summary -- then were reported as
underperforming (`F4 Design Confounds` #4). Every prompt here ends with a
FINAL SUMMARY: delimiter and only what follows it is scored.

A missing delimiter is a parse failure with its own status, never a silent
fallback to the raw text. That is the F1 lesson applied at the other end of the
pipeline.
"""

import pytest

from fairsum.instances import Instance
from fairsum.prompts import (
    PROMPT_BUILDERS,
    FINAL_SUMMARY_MARKER,
    ParseStatus,
    build_repair_prompt,
    parse_final_summary,
)


def make_instance(n=8):
    return Instance(
        instance_id="abc123",
        product_id="lyrica",
        condition_name="fibromyalgia",
        reviews=[
            {"review_id": f"r{i}", "text": f"Review number {i} body.", "product_id": "lyrica"}
            for i in range(n)
        ],
    )


class TestPromptBuilders:
    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_every_prompt_includes_all_review_texts(self, name):
        inst = make_instance()
        prompt = PROMPT_BUILDERS[name](inst)
        for review in inst.reviews:
            assert review["text"] in prompt

    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_a_prompt_mentions_the_marker_exactly_when_its_spec_requires_one(self, name):
        # See test_marker_contract.py: prompts with no reasoning scaffold do not
        # need a marker, and mentioning it only invites a pointless echo.
        from fairsum.prompts import requires_marker

        mentioned = FINAL_SUMMARY_MARKER in PROMPT_BUILDERS[name](make_instance())
        assert mentioned == requires_marker(name)

    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_every_prompt_ends_with_a_length_instruction(self, name):
        assert "60 words" in PROMPT_BUILDERS[name](make_instance())

    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_every_prompt_names_the_indication(self, name):
        assert "fibromyalgia" in PROMPT_BUILDERS[name](make_instance())

    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_every_prompt_states_the_true_review_count(self, name):
        prompt = PROMPT_BUILDERS[name](make_instance(n=6))
        assert "6" in prompt

    def test_the_expected_methods_are_registered(self):
        assert set(PROMPT_BUILDERS) == {
            "direct",
            "refer",
            "ssfp",
            "ecvp",
            "aspect_aware",
            "refer_aspect_aware",
        }

    def test_direct_prompt_does_not_ask_for_perspective_counting(self):
        # The baseline must stay a baseline.
        prompt = PROMPT_BUILDERS["direct"](make_instance()).lower()
        assert "how many" not in prompt


class TestParseFinalSummary:
    def test_extracts_the_text_after_the_marker(self):
        raw = f"Here is my reasoning.\n{FINAL_SUMMARY_MARKER}\nPatients mostly improved."
        result = parse_final_summary(raw)
        assert result.status is ParseStatus.OK
        assert result.summary == "Patients mostly improved."

    def test_uses_the_last_marker_when_the_model_repeats_it(self):
        raw = f"{FINAL_SUMMARY_MARKER} draft\n{FINAL_SUMMARY_MARKER} the real one"
        assert parse_final_summary(raw).summary == "the real one"

    def test_reports_a_missing_marker_instead_of_falling_back(self):
        result = parse_final_summary("Some summary with no marker at all.")
        assert result.status is ParseStatus.NO_MARKER
        assert result.summary is None

    def test_reports_an_empty_summary_after_the_marker(self):
        result = parse_final_summary(f"reasoning\n{FINAL_SUMMARY_MARKER}\n   ")
        assert result.status is ParseStatus.EMPTY
        assert result.summary is None

    def test_marker_matching_ignores_case_and_surrounding_whitespace(self):
        raw = "thinking\n  final summary:  \nThe drug helped most patients."
        result = parse_final_summary(raw)
        assert result.status is ParseStatus.OK
        assert result.summary == "The drug helped most patients."

    def test_strips_trailing_chat_boilerplate_markers(self):
        raw = f"{FINAL_SUMMARY_MARKER}\nPatients improved.\n\n**FINAL SUMMARY**"
        assert parse_final_summary(raw).summary == "Patients improved."


class TestRepairPrompt:
    def test_names_every_missing_event(self):
        prompt = build_repair_prompt(
            make_instance(), draft="Patients improved.", missing_events=["dry skin", "nausea"]
        )
        assert "dry skin" in prompt
        assert "nausea" in prompt

    def test_includes_the_draft_to_revise(self):
        prompt = build_repair_prompt(
            make_instance(), draft="Patients improved.", missing_events=["dry skin"]
        )
        assert "Patients improved." in prompt

    def test_asks_for_the_revised_summary_only(self):
        # Repair returns a summary directly, so it carries no scaffold and needs
        # no marker -- same contract as `direct`.
        prompt = build_repair_prompt(make_instance(), draft="d", missing_events=["x"])
        assert FINAL_SUMMARY_MARKER not in prompt
        assert "summary only" in prompt

    def test_raises_when_there_is_nothing_to_repair(self):
        with pytest.raises(ValueError, match="no missing events"):
            build_repair_prompt(make_instance(), draft="d", missing_events=[])
