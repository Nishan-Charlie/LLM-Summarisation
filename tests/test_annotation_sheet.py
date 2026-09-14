"""Turning a filled annotation spreadsheet into gold annotations.

The sheet is where hand-labelling errors enter, so parsing refuses rather than
guesses. Two refusals matter most:

  * A review not marked done is rejected, never read as "no adverse events".
    `agreement_report` compares only reviews present in both files, so a
    skipped review would silently leave the comparison.
  * Blank certainty or attribution is rejected, never defaulted. Negation and
    condition-attribution are the two failure modes the pilot exists to watch;
    a default would record "asserted, drug" for a decision nobody made.

Every problem is reported at once, so the annotator fixes the sheet in one pass.
"""

import pytest

from fairsum.annotation import (
    AnnotationSheetError,
    Attribution,
    Certainty,
    parse_sheet_rows,
)


def row(review_id="r1", done="yes", **cells):
    return {"review_id": review_id, "done": done, **cells}


def slot(i, span="my skin was dry", event="dry skin", certainty="asserted", attribution="drug"):
    return {
        f"span_{i}": span,
        f"event_{i}": event,
        f"certainty_{i}": certainty,
        f"attribution_{i}": attribution,
    }


def problems_for(rows):
    with pytest.raises(AnnotationSheetError) as exc:
        parse_sheet_rows(rows)
    return exc.value.problems


class TestParseSheetRows:
    def test_a_filled_slot_becomes_an_adverse_event(self):
        [annotation] = parse_sheet_rows([row(**slot(1))])
        [event] = annotation.events
        assert annotation.review_id == "r1"
        assert event.span == "my skin was dry"
        assert event.event == "dry skin"
        assert event.certainty is Certainty.ASSERTED
        assert event.attribution is Attribution.DRUG

    def test_a_done_review_with_no_events_is_an_explicit_empty_annotation(self):
        [annotation] = parse_sheet_rows([row()])
        assert annotation.events == []

    def test_a_review_not_marked_done_is_rejected(self):
        [problem] = problems_for([row(done="")])
        assert "r1" in problem and "done" in problem

    def test_blank_certainty_is_rejected_rather_than_defaulted(self):
        [problem] = problems_for([row(**slot(1, certainty=""))])
        assert "certainty" in problem

    def test_blank_attribution_is_rejected_rather_than_defaulted(self):
        [problem] = problems_for([row(**slot(1, attribution=""))])
        assert "attribution" in problem

    def test_an_invalid_attribution_is_rejected(self):
        [problem] = problems_for([row(**slot(1, attribution="patient"))])
        assert "patient" in problem

    def test_an_event_without_a_span_is_rejected(self):
        [problem] = problems_for([row(**slot(1, span=""))])
        assert "span" in problem

    def test_a_span_without_an_event_is_rejected(self):
        [problem] = problems_for([row(**slot(1, event=""))])
        assert "event" in problem

    def test_fully_blank_slots_are_ignored(self):
        cells = {**slot(1), **slot(2, span="", event="", certainty="", attribution="")}
        [annotation] = parse_sheet_rows([row(**cells)])
        assert len(annotation.events) == 1

    def test_empty_cells_read_as_none_are_treated_as_blank(self):
        # openpyxl returns None for an empty cell.
        cells = slot(1, span=None, event=None, certainty=None, attribution=None)
        [annotation] = parse_sheet_rows([row(**cells)])
        assert annotation.events == []

    def test_event_labels_are_lowercased_and_trimmed(self):
        [annotation] = parse_sheet_rows([row(**slot(1, event="  Dry Skin "))])
        assert annotation.events[0].event == "dry skin"

    def test_dropdown_values_are_case_insensitive(self):
        [annotation] = parse_sheet_rows([row(**slot(1, certainty="Negated"))])
        assert annotation.events[0].certainty is Certainty.NEGATED

    @pytest.mark.parametrize("done", ["yes", "Yes", "y", "x", "TRUE", True, 1])
    def test_done_accepts_common_spellings(self, done):
        [annotation] = parse_sheet_rows([row(done=done)])
        assert annotation.review_id == "r1"

    def test_duplicate_review_ids_are_rejected(self):
        [problem] = problems_for([row("r1"), row("r1")])
        assert "r1" in problem and "duplicate" in problem

    def test_every_problem_is_reported_at_once(self):
        problems = problems_for([row("r1", done=""), row("r2", **slot(1, certainty=""))])
        assert len(problems) == 2
