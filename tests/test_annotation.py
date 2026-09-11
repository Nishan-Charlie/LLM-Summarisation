"""Annotation schema and inter-annotator agreement.

Day 4 of the Week 1 plan is a hard gate: alpha >= 0.7 freezes the schema,
alpha < 0.6 stops the project until the construct is redefined. So the alpha
implementation has to be right, and these tests check it against cases whose
answers are known by hand rather than by running the code.

Negation and attribution are the failure modes the pilot is watching for:
"no side effects at all" and "the pain was from the fibromyalgia, not the drug"
must not become adverse events.
"""

import math

import pytest

from fairsum.annotation import (
    AdverseEvent,
    Attribution,
    Certainty,
    ReviewAnnotation,
    agreement_report,
    krippendorff_alpha,
    normalise_event,
)


class TestAdverseEventSchema:
    def test_an_asserted_drug_attributed_event_counts(self):
        event = AdverseEvent(
            span="my skin was horribly dry",
            event="dry skin",
            certainty=Certainty.ASSERTED,
            attribution=Attribution.DRUG,
        )
        assert event.counts_as_adverse_event is True

    def test_a_negated_event_does_not_count(self):
        event = AdverseEvent(
            span="no side effects at all",
            event="side effects",
            certainty=Certainty.NEGATED,
            attribution=Attribution.DRUG,
        )
        assert event.counts_as_adverse_event is False

    def test_an_event_attributed_to_the_condition_does_not_count(self):
        event = AdverseEvent(
            span="the pain was from the fibromyalgia",
            event="pain",
            certainty=Certainty.ASSERTED,
            attribution=Attribution.CONDITION,
        )
        assert event.counts_as_adverse_event is False

    def test_a_hedged_drug_event_still_counts(self):
        # "might have caused drowsiness" is a reported adverse experience.
        event = AdverseEvent(
            span="might have made me drowsy",
            event="drowsiness",
            certainty=Certainty.HEDGED,
            attribution=Attribution.DRUG,
        )
        assert event.counts_as_adverse_event is True


class TestReviewAnnotation:
    def test_event_set_contains_only_counting_events(self):
        annotation = ReviewAnnotation(
            review_id="r1",
            events=[
                AdverseEvent("dry skin", "dry skin", Certainty.ASSERTED, Attribution.DRUG),
                AdverseEvent("no nausea", "nausea", Certainty.NEGATED, Attribution.DRUG),
            ],
        )
        assert annotation.event_set == {"dry skin"}

    def test_duplicate_events_collapse_to_one(self):
        annotation = ReviewAnnotation(
            review_id="r1",
            events=[
                AdverseEvent("dry skin", "dry skin", Certainty.ASSERTED, Attribution.DRUG),
                AdverseEvent("skin so dry", "dry skin", Certainty.ASSERTED, Attribution.DRUG),
            ],
        )
        assert annotation.event_set == {"dry skin"}

    def test_round_trips_through_a_dict(self):
        annotation = ReviewAnnotation(
            review_id="r1",
            events=[
                AdverseEvent("dry skin", "dry skin", Certainty.ASSERTED, Attribution.DRUG)
            ],
        )
        assert ReviewAnnotation.from_dict(annotation.to_dict()) == annotation


class TestNormaliseEvent:
    def test_maps_a_surface_form_to_its_canonical_label(self):
        table = {"felt like a zombie": "drowsiness"}
        assert normalise_event("Felt Like A Zombie", table) == "drowsiness"

    def test_lowercases_and_strips_unknown_forms(self):
        assert normalise_event("  Dry Skin  ", {}) == "dry skin"


class TestKrippendorffAlpha:
    def test_perfect_agreement_gives_alpha_one(self):
        ratings = {"u1": [1, 1], "u2": [0, 0], "u3": [1, 1]}
        assert krippendorff_alpha(ratings) == pytest.approx(1.0)

    def test_systematic_disagreement_gives_the_hand_computed_value(self):
        # Two units, two coders, each flipping the other. Worked by hand:
        # D_o = 1.0, D_e = 8/12, alpha = 1 - 1.5 = -0.5.
        ratings = {"u1": [1, 0], "u2": [0, 1]}
        assert krippendorff_alpha(ratings) == pytest.approx(-0.5)

    def test_alpha_is_undefined_when_every_rating_is_the_same_value(self):
        # No expected disagreement to normalise against.
        assert math.isnan(krippendorff_alpha({"u1": [1, 1], "u2": [1, 1]}))

    def test_units_rated_by_only_one_coder_are_skipped(self):
        with_singleton = {"u1": [1, 1], "u2": [0, 0], "u3": [1]}
        without = {"u1": [1, 1], "u2": [0, 0]}
        assert krippendorff_alpha(with_singleton) == pytest.approx(
            krippendorff_alpha(without)
        )

    def test_missing_ratings_are_ignored(self):
        assert krippendorff_alpha(
            {"u1": [1, 1, None], "u2": [0, 0, None]}
        ) == pytest.approx(1.0)

    def test_partial_agreement_falls_between_the_extremes(self):
        ratings = {"u1": [1, 1], "u2": [0, 0], "u3": [1, 0], "u4": [0, 0]}
        alpha = krippendorff_alpha(ratings)
        assert -0.5 < alpha < 1.0

    def test_raises_on_no_pairable_units(self):
        with pytest.raises(ValueError, match="pairable"):
            krippendorff_alpha({"u1": [1]})


class TestAgreementReport:
    def test_total_agreement_is_flagged_rather_than_scored_as_alpha_one(self):
        # Every rating is 1, so there is no expected disagreement to normalise
        # against and alpha is genuinely undefined. Coercing it to 1.0 would be
        # the same silent-fallback mistake as v1's degenerate judge rows.
        a = {"r1": {"dry skin"}, "r2": {"nausea"}}
        report = agreement_report(a, dict(a))
        assert math.isnan(report.alpha)
        assert report.perfect_agreement is True
        assert report.n_disagreements == 0

    def test_counts_disagreements_per_event(self):
        a = {"r1": {"dry skin", "nausea"}}
        b = {"r1": {"dry skin"}}
        report = agreement_report(a, b)
        assert report.n_disagreements == 1
        assert ("r1", "nausea") in report.disagreements

    def test_only_reviews_annotated_by_both_are_compared(self):
        a = {"r1": {"dry skin"}, "r2": {"nausea"}}
        b = {"r1": {"dry skin"}}
        assert agreement_report(a, b).n_reviews == 1

    def test_verdict_freezes_the_schema_at_or_above_the_gate(self):
        a = {"r1": {"dry skin"}, "r2": {"nausea"}, "r3": set()}
        assert agreement_report(a, dict(a)).verdict == "freeze"

    def test_verdict_stops_the_project_below_the_lower_gate(self):
        a = {"r1": {"x"}, "r2": set(), "r3": {"y"}, "r4": set()}
        b = {"r1": set(), "r2": {"x"}, "r3": set(), "r4": {"y"}}
        assert agreement_report(a, b).verdict == "stop"
