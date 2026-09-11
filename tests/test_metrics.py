"""Metrics, and above all what happens when the judge fails.

The v1 bug (F1 Degenerate Judge Rows) was that an unparseable judge response
became {label: 0 for all labels}, which normalize_counts turned into a zero
vector, which every metric then scored as maximal unfairness. These tests pin
that behaviour shut: a degenerate count dict must be *reported*, never scored.
"""

import numpy as np
import pytest

from fairsum.metrics import (
    EFFECTIVENESS_LABELS,
    MetricStatus,
    fairsum_metrics,
    normalize_counts,
    spd,
    uer,
)


class TestNormalizeCounts:
    def test_normalizes_counts_to_a_distribution(self):
        counts = {"Ineffective": 2, "Highly Effective": 6}
        dist = normalize_counts(counts, ["Ineffective", "Highly Effective"])
        assert dist == pytest.approx([0.25, 0.75])

    def test_raises_on_all_zero_counts_instead_of_returning_a_zero_vector(self):
        # v1 returned np.zeros(r) here, which silently became "maximally unfair".
        with pytest.raises(ValueError, match="degenerate"):
            normalize_counts({lbl: 0 for lbl in EFFECTIVENESS_LABELS})

    def test_raises_on_empty_counts(self):
        with pytest.raises(ValueError, match="degenerate"):
            normalize_counts({})


class TestMetricIdentity:
    def test_spd_is_exactly_twice_uer_for_genuine_distributions(self):
        # The identity behind the F1 diagnostic: positive and negative
        # deviations are equal when both arguments sum to 1.
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            assert spd(p, q) == pytest.approx(2 * uer(p, q))


class TestFairsumMetrics:
    def test_identical_distributions_are_perfectly_fair(self):
        counts = {"Highly Effective": 4, "Ineffective": 4}
        m = fairsum_metrics(counts, counts)
        assert m.status is MetricStatus.OK
        assert m.spd == pytest.approx(0.0)
        assert m.uer == pytest.approx(0.0)
        assert m.bur == 0
        assert m.auc == pytest.approx(0.0)
        assert m.sof == pytest.approx(0.0)

    def test_flags_a_degenerate_summary_rather_than_scoring_it(self):
        source = {"Highly Effective": 4, "Ineffective": 4}
        m = fairsum_metrics(source, {lbl: 0 for lbl in EFFECTIVENESS_LABELS})
        assert m.status is MetricStatus.DEGENERATE_SUMMARY
        assert m.spd is None
        assert m.uer is None
        assert m.bur is None

    def test_flags_a_degenerate_source(self):
        m = fairsum_metrics({}, {"Highly Effective": 4})
        assert m.status is MetricStatus.DEGENERATE_SOURCE

    def test_bur_fires_when_a_level_falls_below_the_tau_share(self):
        source = {"Highly Effective": 5, "Ineffective": 5}
        summary = {"Highly Effective": 9, "Ineffective": 1}
        m = fairsum_metrics(source, summary, tau=0.8)
        assert m.bur == 1

    def test_bur_does_not_fire_within_tolerance(self):
        source = {"Highly Effective": 5, "Ineffective": 5}
        summary = {"Highly Effective": 6, "Ineffective": 4}  # 0.4 vs 0.8*0.5=0.4
        m = fairsum_metrics(source, summary, tau=0.8)
        assert m.bur == 0

    def test_uer_counts_only_under_representation(self):
        source = {"a": 5, "b": 5}
        summary = {"a": 10, "b": 0}
        m = fairsum_metrics(source, summary, labels=["a", "b"])
        # b is short by 0.5, a is over by 0.5; UER averages only the shortfall.
        assert m.uer == pytest.approx(0.25)

    def test_sof_equals_uer_for_a_binary_attribute(self):
        # F2 Metric Redundancy, pinned as a regression test.
        rng = np.random.default_rng(1)
        for _ in range(20):
            s = {"a": int(rng.integers(1, 9)), "b": int(rng.integers(1, 9))}
            t = {"a": int(rng.integers(0, 9)), "b": int(rng.integers(1, 9))}
            m = fairsum_metrics(s, t, labels=["a", "b"])
            assert m.sof == pytest.approx(m.uer)

    def test_auc_of_a_perfectly_matched_summary_is_zero(self):
        counts = {"a": 3, "b": 5}
        m = fairsum_metrics(counts, counts, labels=["a", "b"])
        assert m.auc == pytest.approx(0.0)

    def test_unknown_labels_in_counts_are_ignored_not_silently_included(self):
        source = {"Highly Effective": 4, "Ineffective": 4}
        summary = {"Highly Effective": 4, "Ineffective": 4, "Bogus Level": 99}
        m = fairsum_metrics(source, summary)
        assert m.status is MetricStatus.OK
        assert m.spd == pytest.approx(0.0)
