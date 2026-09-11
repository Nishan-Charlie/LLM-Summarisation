"""Conformal risk control over verifier scores.

Loss convention: a candidate is emitted when its verifier score clears lambda,
and abstentions contribute zero loss. That makes the risk non-increasing in
lambda, which is what CRC requires -- and it makes the abstention rate a
mandatory companion number, since abstaining on everything satisfies any alpha.

The finite-sample floor 1/(n+1) is the constraint behind `Calibration Design`:
with 45 calibration points no guarantee below ~0.022 is even expressible.
"""

import numpy as np
import pytest

from fairsum.conformal import (
    RiskControlResult,
    abstention_rate,
    empirical_risk,
    finite_sample_floor,
    fit_threshold,
    required_calibration_n,
    risk_coverage_curve,
)


class TestFiniteSampleFloor:
    def test_floor_is_one_over_n_plus_one(self):
        assert finite_sample_floor(45) == pytest.approx(1 / 46)

    def test_required_n_is_the_inverse_of_the_floor(self):
        assert required_calibration_n(0.1) == 9
        assert required_calibration_n(0.05) == 19

    def test_v1_split_cannot_express_a_tight_guarantee(self):
        # 90 instances split in half was the v1-implied calibration size.
        assert finite_sample_floor(45) > 0.02


class TestEmpiricalRisk:
    def test_risk_counts_only_accepted_candidates(self):
        scores = np.array([0.9, 0.5, 0.1])
        losses = np.array([0.0, 1.0, 1.0])
        # lambda = 0.4 accepts the first two; the third abstains and costs 0.
        assert empirical_risk(scores, losses, 0.4) == pytest.approx(1.0 / 3)

    def test_risk_is_zero_when_everything_abstains(self):
        scores = np.array([0.1, 0.2])
        losses = np.array([1.0, 1.0])
        assert empirical_risk(scores, losses, 0.99) == pytest.approx(0.0)

    def test_risk_is_non_increasing_in_lambda(self):
        rng = np.random.default_rng(0)
        scores, losses = rng.random(200), rng.random(200)
        grid = np.linspace(0, 1, 50)
        risks = [empirical_risk(scores, losses, g) for g in grid]
        assert all(a >= b - 1e-12 for a, b in zip(risks, risks[1:]))


class TestAbstentionRate:
    def test_counts_candidates_below_the_threshold(self):
        assert abstention_rate(np.array([0.9, 0.5, 0.1]), 0.4) == pytest.approx(1 / 3)

    def test_is_zero_when_the_threshold_admits_everything(self):
        assert abstention_rate(np.array([0.9, 0.5]), 0.0) == pytest.approx(0.0)


class TestFitThreshold:
    def test_returned_threshold_satisfies_the_bound_on_calibration_data(self):
        rng = np.random.default_rng(1)
        scores = rng.random(300)
        losses = (1 - scores) * rng.random(300)  # better scores, lower loss
        result = fit_threshold(scores, losses, alpha=0.15)
        n = len(scores)
        adjusted = (n * empirical_risk(scores, losses, result.threshold) + 1.0) / (n + 1)
        assert adjusted <= 0.15 + 1e-12

    def test_accepts_everything_when_no_candidate_ever_loses(self):
        scores = np.linspace(0, 1, 100)
        losses = np.zeros(100)
        result = fit_threshold(scores, losses, alpha=0.1)
        assert result.abstention_rate == pytest.approx(0.0)

    def test_forces_abstention_when_every_candidate_loses(self):
        scores = np.linspace(0, 1, 100)
        losses = np.ones(100)
        result = fit_threshold(scores, losses, alpha=0.1)
        assert result.abstention_rate > 0.85

    def test_a_looser_alpha_never_yields_a_stricter_threshold(self):
        rng = np.random.default_rng(2)
        scores = rng.random(300)
        losses = (1 - scores) * rng.random(300)
        loose = fit_threshold(scores, losses, alpha=0.30)
        tight = fit_threshold(scores, losses, alpha=0.05)
        assert loose.threshold <= tight.threshold

    def test_raises_when_alpha_is_below_the_finite_sample_floor(self):
        scores, losses = np.linspace(0, 1, 10), np.zeros(10)
        with pytest.raises(ValueError, match="finite-sample floor"):
            fit_threshold(scores, losses, alpha=0.01)

    def test_rejects_losses_outside_the_unit_interval(self):
        with pytest.raises(ValueError, match="bounded"):
            fit_threshold(np.array([0.5]), np.array([2.0]), alpha=0.5)

    def test_rejects_mismatched_input_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            fit_threshold(np.array([0.5, 0.6]), np.array([0.1]), alpha=0.5)

    def test_result_reports_the_calibration_size(self):
        scores, losses = np.linspace(0, 1, 50), np.zeros(50)
        assert fit_threshold(scores, losses, alpha=0.2).n_calibration == 50


class TestGuaranteeHolds:
    def test_expected_test_risk_respects_alpha_over_repeated_splits(self):
        """The property the whole method rests on.

        CRC bounds the risk in expectation over calibration draws, so the mean
        realised test risk across many exchangeable splits must not exceed alpha.
        """
        rng = np.random.default_rng(3)
        alpha = 0.2
        realised = []
        for _ in range(60):
            scores = rng.random(400)
            losses = (1 - scores) * rng.random(400)
            cal, test = slice(0, 200), slice(200, 400)
            result = fit_threshold(scores[cal], losses[cal], alpha=alpha)
            realised.append(empirical_risk(scores[test], losses[test], result.threshold))
        assert np.mean(realised) <= alpha


class TestRiskCoverageCurve:
    def test_curve_reports_risk_and_abstention_together(self):
        rng = np.random.default_rng(4)
        scores = rng.random(200)
        losses = (1 - scores) * rng.random(200)
        curve = risk_coverage_curve(scores, losses, n_points=10)
        assert list(curve.columns) == ["threshold", "risk", "abstention_rate"]
        assert len(curve) == 10

    def test_abstention_rises_as_risk_falls(self):
        rng = np.random.default_rng(5)
        scores = rng.random(300)
        losses = (1 - scores) * rng.random(300)
        curve = risk_coverage_curve(scores, losses, n_points=20)
        assert curve["abstention_rate"].is_monotonic_increasing
        assert curve["risk"].iloc[0] >= curve["risk"].iloc[-1]


class TestResultDataclass:
    def test_is_serialisable_to_a_plain_dict(self):
        result = RiskControlResult(
            threshold=0.5, alpha=0.1, risk=0.05, abstention_rate=0.2, n_calibration=100
        )
        assert result.to_dict()["threshold"] == 0.5
