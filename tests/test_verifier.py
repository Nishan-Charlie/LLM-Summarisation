"""Adverse-event coverage verification.

Three hard requirements from the vault note `Verifier`, all pinned here:

  1. No silent zero. A verifier that cannot decide returns ABSTAIN, and
     abstentions are counted separately -- never averaged into a score.
  2. Scoring (label-free, drives selection) and measurement (gold labels,
     computes the loss) are separate entry points, so the generator can never
     end up grading itself the way v1's judge did.
  3. Raw model output is retained for every call, so failures are diagnosable
     afterwards rather than inferred from aggregates months later.

The entailment model is injected, so these tests exercise real logic with no GPU.
"""

import pytest

from fairsum.verifier import (
    CoverageReport,
    Verdict,
    coverage_loss,
    measure_coverage,
    score_coverage,
    select_best,
)


class FakeEntailment:
    """Stand-in for the NLI model: returns whatever the fixture dictates."""

    def __init__(self, verdicts, raw="fake-raw"):
        self._verdicts = verdicts
        self._raw = raw
        self.calls = []

    def __call__(self, summary, event):
        self.calls.append((summary, event))
        return self._verdicts.get(event, Verdict.NOT_COVERED), self._raw


class TestMeasureCoverage:
    def test_reports_covered_and_missing_events(self):
        model = FakeEntailment({"dry skin": Verdict.COVERED})
        report = measure_coverage("summary text", ["dry skin", "nausea"], model)
        assert report.covered == ["dry skin"]
        assert report.missing == ["nausea"]

    def test_abstentions_are_tracked_separately_from_omissions(self):
        model = FakeEntailment(
            {"dry skin": Verdict.COVERED, "nausea": Verdict.ABSTAIN}
        )
        report = measure_coverage("s", ["dry skin", "nausea"], model)
        assert report.abstained == ["nausea"]
        assert report.missing == []

    def test_retains_raw_model_output_for_every_event(self):
        model = FakeEntailment({"dry skin": Verdict.COVERED}, raw="entail p=0.91")
        report = measure_coverage("s", ["dry skin", "nausea"], model)
        assert set(report.raw) == {"dry skin", "nausea"}
        assert report.raw["dry skin"] == "entail p=0.91"

    def test_an_instance_with_no_adverse_events_is_not_an_error(self):
        report = measure_coverage("s", [], FakeEntailment({}))
        assert report.covered == [] and report.missing == []

    def test_does_not_call_the_model_when_there_are_no_events(self):
        model = FakeEntailment({})
        measure_coverage("s", [], model)
        assert model.calls == []


class TestCoverageLoss:
    def test_loss_is_the_omitted_fraction(self):
        report = CoverageReport(covered=["a", "b"], missing=["c", "d"], abstained=[])
        assert coverage_loss(report) == pytest.approx(0.5)

    def test_loss_is_zero_when_everything_is_covered(self):
        assert coverage_loss(CoverageReport(covered=["a"], missing=[], abstained=[])) == 0.0

    def test_loss_is_zero_when_there_is_nothing_to_cover(self):
        # No adverse events means nothing could be dropped. Not a division error.
        assert coverage_loss(CoverageReport(covered=[], missing=[], abstained=[])) == 0.0

    def test_abstentions_count_against_coverage_by_default(self):
        # Conservative: an undecidable event is treated as possibly dropped.
        report = CoverageReport(covered=["a"], missing=[], abstained=["b"])
        assert coverage_loss(report) == pytest.approx(0.5)

    def test_abstentions_can_be_excluded_explicitly(self):
        report = CoverageReport(covered=["a"], missing=[], abstained=["b"])
        assert coverage_loss(report, abstain_counts_as_missing=False) == 0.0

    def test_loss_is_bounded_in_the_unit_interval(self):
        # Conformal risk control requires a bounded loss.
        report = CoverageReport(covered=[], missing=["a", "b", "c"], abstained=[])
        assert 0.0 <= coverage_loss(report) <= 1.0


class TestScoreCoverage:
    def test_score_is_the_covered_fraction(self):
        model = FakeEntailment({"a": Verdict.COVERED, "b": Verdict.COVERED})
        assert score_coverage("s", ["a", "b", "c"], model) == pytest.approx(2 / 3)

    def test_score_is_one_when_no_events_are_claimed(self):
        assert score_coverage("s", [], FakeEntailment({})) == pytest.approx(1.0)

    def test_score_is_the_complement_of_the_loss(self):
        model = FakeEntailment({"a": Verdict.COVERED})
        events = ["a", "b"]
        report = measure_coverage("s", events, model)
        assert score_coverage("s", events, model) == pytest.approx(1 - coverage_loss(report))


class TestSelectBest:
    def test_picks_the_highest_scoring_candidate(self):
        best, score, idx = select_best(["a", "b", "c"], [0.1, 0.9, 0.5])
        assert best == "b" and score == pytest.approx(0.9) and idx == 1

    def test_ties_resolve_to_the_first_candidate_deterministically(self):
        _, _, idx = select_best(["a", "b"], [0.7, 0.7])
        assert idx == 0

    def test_raises_on_an_empty_candidate_set(self):
        with pytest.raises(ValueError, match="at least one candidate"):
            select_best([], [])

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            select_best(["a", "b"], [0.5])
