"""Day 5 -- how many calibration instances does the guarantee actually need?

Two questions, the second more consequential than the first:

  1. What is the finite-sample floor at a given n? Below it, an alpha simply
     cannot be expressed no matter how good the model is.
  2. Does effective n read as the instance count or the DRUG count? Instances
     built from a shared review pool are not independent, so if the answer is
     drugs then the annotation budget must widen (more drugs, fewer reviews
     each) rather than deepen -- and that changes the roadmap this week rather
     than in Week 6.

Question 2 is simulated by resampling calibration sets two ways: by instance
(treating them as independent, the optimistic reading) and by drug (carrying all
of a drug's instances together, the honest one). If the realised-risk spread is
much wider under drug resampling, drugs are the real unit.

Usage:
    python scripts/power_analysis.py --alpha 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fairsum.conformal import (
    empirical_risk,
    finite_sample_floor,
    fit_threshold,
    required_calibration_n,
)


def floor_table(alphas=(0.05, 0.1, 0.2)) -> pd.DataFrame:
    """Feasibility is the wrong question; the useful one is how much of the alpha
    budget the finite-sample slack consumes.

    CRC accepts a threshold when (n * R + B) / (n + 1) <= alpha, so the empirical
    risk may only reach roughly alpha - B/(n+1). At n=45 and alpha=0.1 that means
    22% of the budget is spent on slack rather than on tolerating real omissions.
    """
    rows = []
    for n in (45, 90, 150, 200, 300, 500):
        row = {"n_calibration": n, "floor": finite_sample_floor(n)}
        for alpha in alphas:
            if alpha < finite_sample_floor(n):
                row[f"slack%@a={alpha}"] = float("nan")
            else:
                row[f"slack%@a={alpha}"] = 100 * finite_sample_floor(n) / alpha
        rows.append(row)
    return pd.DataFrame(rows)


def simulate(
    alpha: float,
    n_drugs: int,
    instances_per_drug: int,
    n_trials: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare risk spread when resampling by instance versus by drug.

    Instances of one drug share reviews, so their scores and losses are
    correlated. That correlation is what makes the instance count an
    overstatement of the effective sample size.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_trials):
        # Drug-level effect plus instance noise: the correlation structure the
        # shared review pool actually induces.
        drug_effect = rng.normal(0, 0.25, size=n_drugs)
        drug_ids = np.repeat(np.arange(n_drugs), instances_per_drug)
        scores = np.clip(
            rng.normal(0.6, 0.2, size=len(drug_ids)) + drug_effect[drug_ids], 0, 1
        )
        losses = np.clip(
            (1 - scores) * rng.random(len(drug_ids)) + 0.1 * drug_effect[drug_ids], 0, 1
        )

        n_total = len(drug_ids)
        half_drugs = n_drugs // 2

        # (a) resample by instance -- assumes independence
        perm = rng.permutation(n_total)
        cal_i, test_i = perm[: n_total // 2], perm[n_total // 2 :]
        result_i = fit_threshold(scores[cal_i], losses[cal_i], alpha=alpha)
        rows.append({
            "unit": "instance",
            "realised_risk": empirical_risk(scores[test_i], losses[test_i], result_i.threshold),
            "abstention": result_i.abstention_rate,
        })

        # (b) resample by drug -- carries a drug's instances together
        drug_perm = rng.permutation(n_drugs)
        cal_drugs = set(drug_perm[:half_drugs].tolist())
        mask = np.array([d in cal_drugs for d in drug_ids])
        result_d = fit_threshold(scores[mask], losses[mask], alpha=alpha)
        rows.append({
            "unit": "drug",
            "realised_risk": empirical_risk(scores[~mask], losses[~mask], result_d.threshold),
            "abstention": result_d.abstention_rate,
        })

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--drugs", type=int, default=53, help="drugs available (v1 had 53)")
    parser.add_argument("--per-drug", type=int, default=8)
    parser.add_argument("--trials", type=int, default=300)
    args = parser.parse_args()

    print("=" * 66)
    print("1. FINITE-SAMPLE FLOOR")
    print("=" * 66)
    print(floor_table().to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nalpha={args.alpha} needs at least n={required_calibration_n(args.alpha)} "
          f"calibration points to be expressible at all.")
    print("v1's implied split (90 instances, half calibration) gives a floor of "
          f"{finite_sample_floor(45):.4f} -- a fifth of a 0.1 target.")

    print("\n" + "=" * 66)
    print("2. IS THE EFFECTIVE UNIT THE INSTANCE OR THE DRUG?")
    print("=" * 66)
    sim = simulate(args.alpha, args.drugs, args.per_drug, args.trials)
    summary = sim.groupby("unit")["realised_risk"].agg(
        mean="mean", sd="std", q05=lambda s: s.quantile(0.05),
        q95=lambda s: s.quantile(0.95), exceed=lambda s: (s > args.alpha).mean(),
    )
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n`exceed` = share of trials whose realised risk exceeded alpha={args.alpha}.")

    ratio = summary.loc["drug", "sd"] / max(summary.loc["instance", "sd"], 1e-9)
    print(f"\nrisk-spread ratio (drug / instance): {ratio:.2f}x")
    if ratio > 1.5:
        print(
            "\nVERDICT: drug-level resampling is materially noisier, so the effective\n"
            "sample size tracks the DRUG count, not the instance count. Widen the\n"
            "annotation pool across MORE DRUGS rather than more reviews per drug,\n"
            "and report drug counts beside instance counts in every table."
        )
    else:
        print(
            "\nVERDICT: the two are comparable at this correlation level. Instance\n"
            "count is a defensible n, but still report drug counts alongside."
        )

    out = Path("results/day05_power.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out)
    print(f"\nSaved: {out}")
    print("\nNOTE: the correlation structure here is assumed, not measured. Re-run "
          "with real scores once Week 2 has generations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
