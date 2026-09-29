"""The Day 4 gate statistic.

Discovered while wiring up the annotation sheet: `agreement_report` computed
Krippendorff's alpha over (review, event) pairs that at least one annotator
marked. No pair is ever "both absent", so with a agreements and d disagreements
alpha = (1 - d) / (2a + d) -- zero or below whenever a single disagreement
exists. A pilot with 99% agreement returned alpha = 0.0 and the verdict STOP.
The old tests covered only perfect agreement and total disagreement, the two
cases that formula happens to get right.

Re-registered 2026-09-14, before any annotation existed: pairwise F1 gates
(freeze >= 0.8, revise 0.7-0.8, stop < 0.7), the standard measure when true
negatives are undefined (Hripcsak & Rothschild, JAMIA 2005). Krippendorff's
alpha under MASI distance, one unit per review (Passonneau, LREC 2006), is
reported beside it as the chance-corrected figure.
"""

import math

import pytest

from fairsum.annotation import (
    F1_FREEZE_THRESHOLD,
    F1_STOP_THRESHOLD,
    agreement_report,
    krippendorff_alpha,
    krippendorff_alpha_distance,
    masi_distance,
    pairwise_f1,
)


def pilot(n_agree, n_disagree):
    """n_agree reviews where both mark the same event, n_disagree where only A does."""
    a, b = {}, {}
    for i in range(n_agree):
        a[f"r{i}"] = {f"e{i}"}
        b[f"r{i}"] = {f"e{i}"}
    for j in range(n_disagree):
        a[f"d{j}"] = {f"x{j}"}
        b[f"d{j}"] = set()
    return a, b


class TestTheRegression:
    def test_ninety_nine_percent_agreement_does_not_stop_the_project(self):
        assert agreement_report(*pilot(100, 1)).verdict == "freeze"

    def test_realistic_agreement_is_not_scored_at_or_below_zero(self):
        report = agreement_report(*pilot(120, 20))
        assert report.f1 > 0.8
        assert report.alpha > 0


class TestPairwiseF1:
    def test_is_twice_the_shared_events_over_all_marked_events(self):
        # A marks {x, y}, B marks {x}: 2*1 / (2+1).
        assert pairwise_f1({"r1": {"x", "y"}}, {"r1": {"x"}}) == pytest.approx(2 / 3)

    def test_is_symmetric(self):
        a = {"r1": {"x", "y"}, "r2": {"z"}}
        b = {"r1": {"x"}, "r2": {"z", "w"}}
        assert pairwise_f1(a, b) == pytest.approx(pairwise_f1(b, a))

    def test_is_undefined_when_nobody_marked_anything(self):
        assert math.isnan(pairwise_f1({"r1": set()}, {"r1": set()}))

    def test_ignores_reviews_only_one_annotator_labelled(self):
        assert pairwise_f1({"r1": {"x"}, "r2": {"y"}}, {"r1": {"x"}}) == pytest.approx(1.0)


class TestGateThresholds:
    def test_the_registered_thresholds(self):
        assert (F1_FREEZE_THRESHOLD, F1_STOP_THRESHOLD) == (0.8, 0.7)

    def test_exactly_at_the_freeze_threshold_freezes(self):
        # Four shared events and one extra on each side: 2*4 / (5+5) = 0.8.
        a = {"r1": {"a", "b", "c", "d", "e"}}
        b = {"r1": {"a", "b", "c", "d", "f"}}
        report = agreement_report(a, b)
        assert report.f1 == pytest.approx(0.8)
        assert report.verdict == "freeze"

    def test_between_the_thresholds_revises(self):
        # 2*3 / (4+4) = 0.75.
        a = {"r1": {"a", "b", "c", "d"}}
        b = {"r1": {"a", "b", "c", "e"}}
        assert agreement_report(a, b).verdict == "revise"

    def test_below_the_stop_threshold_stops(self):
        # 2*2 / (4+4) = 0.5.
        a = {"r1": {"a", "b", "c", "d"}}
        b = {"r1": {"a", "b", "e", "f"}}
        assert agreement_report(a, b).verdict == "stop"

    def test_no_marked_events_is_undefined_not_a_pass(self):
        assert agreement_report({"r1": set()}, {"r1": set()}).verdict == "undefined"


class TestMasiDistance:
    def test_identical_sets_are_zero(self):
        assert masi_distance({"a", "b"}, {"b", "a"}) == 0.0

    def test_two_empty_sets_agree(self):
        # Both annotators saying a review reports no adverse events is agreement.
        assert masi_distance(set(), set()) == 0.0

    def test_disjoint_sets_are_one(self):
        assert masi_distance({"a"}, {"b"}) == 1.0

    def test_empty_against_non_empty_is_one(self):
        assert masi_distance(set(), {"a"}) == 1.0

    def test_containment_is_weighted_two_thirds(self):
        # Jaccard 1/2, weight 2/3: 1 - 1/3.
        assert masi_distance({"a"}, {"a", "b"}) == pytest.approx(2 / 3)

    def test_overlap_without_containment_is_weighted_one_third(self):
        # Jaccard 1/3, weight 1/3: 1 - 1/9.
        assert masi_distance({"a", "b"}, {"b", "c"}) == pytest.approx(8 / 9)


def nominal(x, y):
    return 0.0 if x == y else 1.0


class TestKrippendorffAlphaDistance:
    @pytest.mark.parametrize(
        "ratings",
        [
            {"u1": [1, 1], "u2": [0, 0], "u3": [1, 0], "u4": [0, 0]},
            {"u1": [1, 0], "u2": [0, 1]},
            {"u1": [1, 1], "u2": [0, 0], "u3": [1]},
        ],
    )
    def test_a_zero_one_distance_reproduces_nominal_alpha(self, ratings):
        assert krippendorff_alpha_distance(ratings, nominal) == pytest.approx(
            krippendorff_alpha(ratings)
        )

    def test_masi_alpha_matches_a_hand_computed_value(self):
        # Values {a,b}, {a}, {c}, {c}; n = 4.
        # D_o = (1/4) * (2 * 2/3)          = 1/3
        # D_e = (1/12) * (2 * 2/3 + 8 * 1) = 7/9
        # alpha = 1 - (1/3) / (7/9)        = 4/7
        ratings = {
            "u1": [frozenset({"a", "b"}), frozenset({"a"})],
            "u2": [frozenset({"c"}), frozenset({"c"})],
        }
        assert krippendorff_alpha_distance(ratings, masi_distance) == pytest.approx(4 / 7)

    def test_is_undefined_when_every_value_is_identical(self):
        ratings = {"u1": [frozenset(), frozenset()], "u2": [frozenset(), frozenset()]}
        assert math.isnan(krippendorff_alpha_distance(ratings, masi_distance))

    def test_raises_on_no_pairable_units(self):
        with pytest.raises(ValueError, match="pairable"):
            krippendorff_alpha_distance({"u1": [frozenset()]}, masi_distance)


class TestReportCarriesBothStatistics:
    def test_summary_shows_the_gate_and_the_alpha(self):
        summary = agreement_report(*pilot(10, 2)).summary()
        assert "F1 (gate)" in summary and "MASI alpha" in summary

    def test_alpha_counts_reviews_both_left_empty_as_agreement(self):
        # F1 cannot see a review both annotators left empty; MASI alpha can.
        # Hand values: 6/11 with the empty review, 0.4 without it.
        a = {"r1": {"x"}, "r2": set(), "r3": {"y"}}
        b = {"r1": {"x"}, "r2": set(), "r3": set()}
        with_empty = agreement_report(a, b).alpha
        without = agreement_report({"r1": {"x"}, "r3": {"y"}}, {"r1": {"x"}, "r3": set()}).alpha
        assert with_empty == pytest.approx(6 / 11)
        assert without == pytest.approx(0.4)
