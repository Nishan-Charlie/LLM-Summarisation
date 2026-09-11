"""Day 1 -- is v1 measuring fairness, or judge parse failures?

Compares the degenerate-row rate measured directly against the rate implied by
the published aggregates, then recomputes the method table with degenerate rows
excluded to see whether the reported ranking survives.

Prediction on record (vault note `F1 Degenerate Judge Rows`): overall f ~ 0.26,
varying 7x across methods, and `direct` overtakes `ssfp` on SPD once the
degenerate rows are removed.

Usage:
    python scripts/run_audit.py --csv data/generated_summaries_with_fairness_metrics.csv

The CSV is the one v1 wrote to
`MyDrive/drug_summarization_method_A_full_outputs/`; download it first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fairsum.audit import audit_results, implied_degenerate_fraction

# The v1 method_table, transcribed from cell 24 of drug_summary_01_Method_A.ipynb.
V1_PUBLISHED = {
    "direct": (0.1761, 0.1358),
    "drug_aspect_aware": (0.1709, 0.1255),
    "ecvp": (0.2004, 0.1124),
    "refer": (0.1857, 0.1162),
    "refer_drug_aspect_aware": (0.2152, 0.1332),
    "ssfp": (0.1997, 0.1065),
}


def predicted_table(n_labels: int = 5) -> pd.DataFrame:
    rows = [
        {
            "method": method,
            "SPD": spd,
            "UER": uer,
            "implied_degenerate": implied_degenerate_fraction(spd, uer, n_labels),
        }
        for method, (spd, uer) in V1_PUBLISHED.items()
    ]
    return pd.DataFrame(rows).sort_values("implied_degenerate", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="v1 results CSV with summary_counts")
    parser.add_argument("--labels", type=int, default=5)
    args = parser.parse_args()

    predicted = predicted_table(args.labels)
    print("=" * 62)
    print("PREDICTED from v1 published aggregates  (no data needed)")
    print("=" * 62)
    print(predicted.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nmean implied degenerate rate: {predicted['implied_degenerate'].mean():.4f}")

    if not args.csv:
        print("\nPass --csv to measure the actual rate and settle it.")
        return 0

    if not args.csv.exists():
        print(f"\nERROR: {args.csv} not found.", file=sys.stderr)
        print(
            "Download `generated_summaries_with_fairness_metrics.csv` from "
            "MyDrive/drug_summarization_method_A_full_outputs/.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(args.csv)
    report = audit_results(df)
    print("\n" + report.summary())

    measured = report.by_method["degenerate_rate"]
    comparison = predicted.set_index("method").join(
        measured.rename("measured_degenerate")
    )
    print("\n" + "=" * 62)
    print("PREDICTED vs MEASURED")
    print("=" * 62)
    print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))

    if report.clean_tables is not None:
        print("\n" + "=" * 62)
        print("METHOD TABLE with degenerate rows EXCLUDED")
        print("=" * 62)
        print(report.clean_tables.to_string())
        if "SPD" in report.clean_tables.columns:
            best = report.clean_tables["SPD"].idxmin()
            print(f"\nBest method on SPD after cleaning: {best}")
            print("v1 reported `direct` best on SPD *including* degenerate rows.")

    out = Path("results/day01_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
