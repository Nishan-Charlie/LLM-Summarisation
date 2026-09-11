"""Conformal risk control over verifier scores (Angelopoulos et al., ICLR 2024).

A candidate summary is emitted when its verifier score clears `lambda`, and
abstains otherwise. Abstentions contribute zero loss, which makes the empirical
risk non-increasing in `lambda` -- the monotonicity CRC needs.

    lambda_hat = inf { lambda : (n * R_hat(lambda) + B) / (n + 1) <= alpha }

with B = 1 the loss bound. Taking the infimum picks the most permissive
threshold that still satisfies the bound, i.e. the fewest abstentions.

WHAT THE GUARANTEE SAYS: on a fresh exchangeable instance, the expected
adverse-event omission of a non-abstained summary is at most `alpha`.
WHAT IT DOES NOT SAY: anything about an individual summary (the bound is
marginal), and nothing at all under distribution shift. See the vault note
`Conformal Risk Control`.

Abstaining on everything satisfies any alpha, so `abstention_rate` must be
reported beside every risk number. `RiskControlResult` carries both so they
cannot be separated by accident.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LOSS_BOUND = 1.0


def finite_sample_floor(n_calibration: int) -> float:
    """Smallest alpha expressible with `n` calibration points: B / (n + 1).

    Even with total abstention the adjusted risk cannot fall below this, so an
    alpha underneath it is infeasible regardless of how good the model is.
    """
    return LOSS_BOUND / (n_calibration + 1)


def required_calibration_n(alpha: float) -> int:
    """Minimum calibration size for `alpha` to be attainable at all."""
    return math.ceil(LOSS_BOUND / alpha) - 1


def empirical_risk(scores: np.ndarray, losses: np.ndarray, threshold: float) -> float:
    """Mean loss with abstentions (score below `threshold`) counted as zero."""
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if scores.size == 0:
        return 0.0
    return float(np.mean(np.where(scores >= threshold, losses, 0.0)))


def abstention_rate(scores: np.ndarray, threshold: float) -> float:
    """Share of instances whose best candidate fails to clear `threshold`."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores < threshold))


@dataclass(frozen=True)
class RiskControlResult:
    threshold: float
    alpha: float
    risk: float
    abstention_rate: float
    n_calibration: int

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"lambda={self.threshold:.4f}  alpha={self.alpha:.3f}  "
            f"calibration risk={self.risk:.4f}  "
            f"abstention={self.abstention_rate:.1%}  n={self.n_calibration}"
        )


def _candidate_grid(scores: np.ndarray) -> np.ndarray:
    """Thresholds worth testing: just above each observed score, plus a floor.

    Adding a tiny epsilon above each score makes `>=` flip exactly at the
    observed values, so the grid covers every distinct accept/reject partition.
    """
    unique = np.unique(scores)
    eps = 1e-9
    return np.concatenate([[-np.inf], unique, unique + eps, [np.inf]])


def fit_threshold(
    scores: np.ndarray,
    losses: np.ndarray,
    alpha: float,
) -> RiskControlResult:
    """Calibrate the acceptance threshold so expected loss is at most `alpha`.

    `scores` are label-free verifier scores (higher is better) for the selected
    candidate of each calibration instance; `losses` are the gold adverse-event
    omission fractions for those same candidates.
    """
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)

    if scores.shape != losses.shape:
        raise ValueError(
            f"scores and losses must have the same length "
            f"(got {scores.shape} and {losses.shape})"
        )
    if losses.size and (losses.min() < 0.0 or losses.max() > LOSS_BOUND):
        raise ValueError(
            f"losses must be bounded in [0, {LOSS_BOUND}]; "
            f"got [{losses.min()}, {losses.max()}]"
        )

    n = len(scores)
    floor = finite_sample_floor(n)
    if alpha < floor:
        raise ValueError(
            f"alpha={alpha} is below the finite-sample floor {floor:.4f} for "
            f"n={n} calibration points. Need at least "
            f"n={required_calibration_n(alpha)} to express this guarantee."
        )

    for threshold in _candidate_grid(scores):
        risk = empirical_risk(scores, losses, threshold)
        if (n * risk + LOSS_BOUND) / (n + 1) <= alpha:
            return RiskControlResult(
                threshold=float(threshold),
                alpha=alpha,
                risk=risk,
                abstention_rate=abstention_rate(scores, threshold),
                n_calibration=n,
            )

    # Unreachable: the +inf threshold abstains on everything, giving risk 0,
    # and alpha >= floor was checked above.
    raise AssertionError("no threshold satisfied the bound despite alpha >= floor")


def risk_coverage_curve(
    scores: np.ndarray,
    losses: np.ndarray,
    n_points: int = 25,
) -> pd.DataFrame:
    """Risk against abstention across the threshold range.

    The paper's primary figure: a method is only better if it lowers risk at a
    matched abstention rate. Reporting either number alone is meaningless.
    """
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)
    grid = np.linspace(float(scores.min()), float(scores.max()) + 1e-9, n_points)
    return pd.DataFrame(
        {
            "threshold": grid,
            "risk": [empirical_risk(scores, losses, t) for t in grid],
            "abstention_rate": [abstention_rate(scores, t) for t in grid],
        }
    )
