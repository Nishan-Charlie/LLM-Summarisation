"""Audit of the v1 notebook outputs -- Day 1 of the Week 1 plan.

The question: are the reported fairness numbers measuring summaries, or are they
measuring how often the judge failed to emit parseable JSON?

Background in the vault note `F1 Degenerate Judge Rows`. The short version:
v1's `classify_summary_counts` returned all-zero counts on a parse failure, and
`normalize_counts` turned that into a zero vector, so every such row scored
SPD=UER=1/r, BUR=1, AUC=1.0 -- indistinguishable from a genuinely unfair summary.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

# Below this, degenerate rows are noise; above it, they drive the aggregates.
MATERIAL_DEGENERATE_RATE = 0.02


def parse_counts(value: Any) -> dict[str, float]:
    """Read a counts column that may be a dict, JSON, or a Python repr.

    v1 wrote some columns with `json.dumps` and others with `str(dict)`, so both
    quoting styles appear. Unparseable values become `{}`, which downstream code
    treats as degenerate rather than as zero counts.
    """
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return {}
    for loads in (json.loads, ast.literal_eval):
        try:
            parsed = loads(value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _is_degenerate(counts: Any) -> bool:
    parsed = parse_counts(counts)
    return sum(float(v) for v in parsed.values()) <= 0


def degenerate_fraction(rows: Iterable[Any]) -> float:
    """Share of count dicts that carry no mass at all."""
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(_is_degenerate(r) for r in rows) / len(rows)


def implied_degenerate_fraction(
    mean_spd: float,
    mean_uer: float,
    n_labels: int,
) -> float:
    """Back the degenerate rate out of published aggregates alone.

    For genuine distributions `spd == 2 * uer`; for a degenerate row both equal
    `1 / n_labels`. Mixing a fraction `f` of degenerate rows therefore gives

        mean_spd = f/r + (1-f) * 2u
        mean_uer = f/r + (1-f) * u
        =>  f = r * (2 * mean_uer - mean_spd)

    This is what let the hypothesis be formed from the notebook's printed tables
    before any code was re-run. Clipped to [0, 1]: values outside that range mean
    the mixture model does not hold, and something other than degenerate rows is
    breaking the identity.
    """
    f = n_labels * (2.0 * mean_uer - mean_spd)
    return float(min(1.0, max(0.0, f)))


def identity_residuals(df: pd.DataFrame, spd_col: str = "SPD", uer_col: str = "UER") -> pd.Series:
    """Per-row `spd - 2 * uer`. Zero for valid rows, `-1/r` for degenerate ones.

    A residual that is neither tells you the metric code is broken in some way
    the degenerate-row story does not explain -- worth knowing before rebuilding.
    """
    return df[spd_col] - 2.0 * df[uer_col]


@dataclass
class AuditReport:
    overall_degenerate_rate: float
    by_method: pd.DataFrame
    rate_varies_by_method: bool
    verdict: str
    clean_tables: pd.DataFrame | None = None

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "V1 AUDIT -- degenerate judge rows",
            "=" * 62,
            f"overall degenerate rate : {self.overall_degenerate_rate:.4f}",
            f"varies across methods   : {self.rate_varies_by_method}",
            f"verdict                 : {self.verdict.upper()}",
            "",
            self.by_method.to_string(),
        ]
        if self.verdict == "void":
            lines += [
                "",
                "Degenerate rows are scored as maximally unfair, so every v1 table",
                "mixes judge failures into the fairness aggregates. Regenerate or",
                "discard. See the vault note `F1 Degenerate Judge Rows`.",
            ]
        return "\n".join(lines)


def audit_results(
    df: pd.DataFrame,
    counts_col: str = "summary_counts",
    method_col: str = "method",
) -> AuditReport:
    """Measure the degenerate rate overall and per method, and give a verdict."""
    flags = df[counts_col].map(_is_degenerate)
    overall = float(flags.mean()) if len(df) else 0.0

    by_method = (
        pd.DataFrame({method_col: df[method_col], "degenerate": flags})
        .groupby(method_col)["degenerate"]
        .agg(degenerate_rate="mean", n="size")
        .sort_values("degenerate_rate", ascending=False)
    )

    rates = by_method["degenerate_rate"]
    varies = bool(len(rates) > 1 and (rates.max() - rates.min()) > 1e-9)

    verdict = "void" if overall > MATERIAL_DEGENERATE_RATE else "clean"

    clean_tables = None
    metric_cols = [c for c in ("SPD", "BUR", "UER", "SOF", "AUC") if c in df.columns]
    if metric_cols and overall > 0:
        clean_tables = (
            df.loc[~flags]
            .groupby(method_col)[metric_cols]
            .mean()
            .round(4)
        )

    return AuditReport(
        overall_degenerate_rate=overall,
        by_method=by_method,
        rate_varies_by_method=varies,
        verdict=verdict,
        clean_tables=clean_tables,
    )
