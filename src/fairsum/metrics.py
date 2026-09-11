"""FairSumm metrics (Zhang et al. 2024, Definitions 5.1-5.4).

Reimplemented with one behavioural change from the v1 notebooks: a count dict
that carries no mass is a *measurement failure*, not a maximally unfair summary.
v1 mapped it to a zero vector, which made SPD=UER=0.2, BUR=1 and AUC=1.0 for
every unparseable judge response. See the vault note `F1 Degenerate Judge Rows`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

EFFECTIVENESS_LABELS: list[str] = [
    "Ineffective",
    "Marginally Effective",
    "Moderately Effective",
    "Considerably Effective",
    "Highly Effective",
]

DEFAULT_TAU = 0.8
DEFAULT_AUC_POINTS = 10


class MetricStatus(Enum):
    """Why a row does or does not carry a score."""

    OK = "ok"
    DEGENERATE_SOURCE = "degenerate_source"
    DEGENERATE_SUMMARY = "degenerate_summary"


@dataclass(frozen=True)
class Metrics:
    """Metric values, or `None` for every field when `status` is not OK.

    Callers must branch on `status`. There is deliberately no numeric fallback:
    averaging a failure into a score is the v1 bug.
    """

    status: MetricStatus
    spd: float | None = None
    bur: int | None = None
    uer: float | None = None
    auc: float | None = None
    sof: float | None = None


def normalize_counts(
    counts: Mapping[str, float],
    labels: Sequence[str] = EFFECTIVENESS_LABELS,
) -> np.ndarray:
    """Definition 3.1 -- raw value counts to a distribution over `labels`.

    Counts for keys outside `labels` are ignored. Raises if no mass remains.
    """
    values = np.array([float(counts.get(label, 0)) for label in labels])
    total = values.sum()
    if total <= 0:
        raise ValueError(
            "degenerate count vector: no mass over the given labels "
            f"(labels={list(labels)}, counts={dict(counts)})"
        )
    return values / total


def spd(source: np.ndarray, summary: np.ndarray) -> float:
    """Mean absolute difference between the two distributions. Lower is fairer.

    Note `spd == 2 * uer` whenever both arguments are genuine distributions;
    `fairsum.audit` uses that identity to detect degenerate rows in v1 output.
    """
    return float(np.mean(np.abs(source - summary)))


def bur(source: np.ndarray, summary: np.ndarray, tau: float = DEFAULT_TAU) -> int:
    """Definition 5.1 -- unfair (1) iff any value falls below its tau share."""
    return int(np.any(summary < tau * source))


def uer(source: np.ndarray, summary: np.ndarray) -> float:
    """Definition 5.2 -- mean under-representation, as a raw proportion."""
    return float(np.mean(np.maximum(0.0, source - summary)))


def auc(
    source: np.ndarray,
    summary: np.ndarray,
    n_points: int = DEFAULT_AUC_POINTS,
) -> float:
    """Definition 5.3 -- BUR integrated over tau, on an evenly spaced grid.

    Lower is fairer. Unrelated to ROC-AUC despite the name inherited from the
    paper; rename before drafting (vault note `F3 BUR Saturation`).
    """
    taus = [(i + 0.5) / n_points for i in range(n_points)]
    return float(np.mean([bur(source, summary, tau=t) for t in taus]))


def sof(source: np.ndarray, summary: np.ndarray) -> float:
    """Definition 5.4 -- mean absolute deviation of the shortfall set.

    Mean absolute deviation, not standard deviation. Degenerates to `uer` for a
    binary attribute (vault note `F2 Metric Redundancy`).
    """
    shortfall = np.maximum(0.0, source - summary)
    return float(np.mean(np.abs(shortfall - shortfall.mean())))


def fairsum_metrics(
    source_counts: Mapping[str, float],
    summary_counts: Mapping[str, float],
    labels: Sequence[str] = EFFECTIVENESS_LABELS,
    tau: float = DEFAULT_TAU,
    n_points: int = DEFAULT_AUC_POINTS,
) -> Metrics:
    """All five metrics for one (source, summary) pair.

    Returns a `Metrics` whose `status` says whether the numbers are meaningful.
    A degenerate side yields `status != OK` and no values at all.
    """
    try:
        source_dist = normalize_counts(source_counts, labels)
    except ValueError:
        return Metrics(status=MetricStatus.DEGENERATE_SOURCE)

    try:
        summary_dist = normalize_counts(summary_counts, labels)
    except ValueError:
        return Metrics(status=MetricStatus.DEGENERATE_SUMMARY)

    return Metrics(
        status=MetricStatus.OK,
        spd=spd(source_dist, summary_dist),
        bur=bur(source_dist, summary_dist, tau=tau),
        uer=uer(source_dist, summary_dist),
        auc=auc(source_dist, summary_dist, n_points=n_points),
        sof=sof(source_dist, summary_dist),
    )
