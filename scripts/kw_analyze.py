"""Tables and figures from scores (spec §2 and §5).

    python scripts/kw_analyze.py --config configs/kw_week1.toml

Writes results/kw/<run>/tables/*.csv and results/kw/<run>/figures/*.png.
"""

from __future__ import annotations

import argparse

import pandas as pd

from fairsum.kwsum.compliance import SCORED_STATUSES
from fairsum.kwsum.config import load_config
from fairsum.kwsum.figures import fig_majority_amplification, fig_tradeoff
from fairsum.kwsum.scoring import SCORER_VERSION
from fairsum.kwsum.stats import (
    compliance_table, condition_means, per_level_effects, polarity_only_view, run_tests,
    worst_case_fill,
)
from fairsum.kwsum.store import read_jsonl

PRIMARY = ["unsupported_rate", "contradiction_rate", "count_abs_error", "delta_p",
           "majority_amplification", "coverage_gap"]
SECONDARY = ["support_score", "recall_minority", "attribution_share_minority", "bur", "uer",
             "keyword_uptake", "keyword_uptake_minority", "invented_numbers", "n_words"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    scores = pd.DataFrame(read_jsonl(cfg.scores_path))
    if scores.empty:
        raise SystemExit(f"no scores at {cfg.scores_path}; run kw_score.py first")
    current = scores.get("scorer_version", pd.Series(index=scores.index, dtype=object)) == SCORER_VERSION
    print(f"using {int(current.sum())} rows scored by v{SCORER_VERSION}; "
          f"dropping {int((~current).sum())} stale rows")
    scores = scores[current].copy()
    tables, figures = cfg.results_dir / "tables", cfg.results_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    compliance_table(scores).to_csv(tables / "compliance.csv", index=False)
    condition_means(scores, PRIMARY + SECONDARY).to_csv(tables / "means.csv", index=False)
    condition_means(scores, PRIMARY, by=("model", "dataset", "level", "condition")).to_csv(
        tables / "means_by_level.csv", index=False)
    condition_means(worst_case_fill(scores), ["unsupported_rate", "contradiction_rate", "delta_p"],
                    statuses=SCORED_STATUSES | {"worst_case"}).to_csv(
        tables / "means_worst_case.csv", index=False)
    tests = run_tests(scores)
    tests.to_csv(tables / "hypotheses.csv", index=False)
    per_level_effects(scores).to_csv(tables / "hypotheses_by_level_descriptive.csv", index=False)
    # Robustness: the same registered tests with proportions from the polarity model alone.
    run_tests(polarity_only_view(scores)).to_csv(
        tables / "hypotheses_robustness_polarity_only.csv", index=False)
    manip = (scores.drop_duplicates("collection_id")
             .groupby("dataset")["negation_changed"].mean().rename("share_negation_changed"))
    manip.to_csv(tables / "negation_manipulation_rate.csv")
    print("H3 manipulation rate (collections where negation changed the prompt):",
          manip.round(3).to_dict())

    fig_majority_amplification(scores, figures / "majority_amplification.png")
    fig_tradeoff(scores, figures / "faithfulness_fairness_tradeoff.png")

    cols = ["model", "dataset", "hypothesis", "comparison", "metric", "n",
            "mean_diff", "ci_lo", "ci_hi", "p_holm", "supported"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(tests[cols].round(4).to_string(index=False) if not tests.empty else "no tests ran")
    print(f"tables in {tables}\nfigures in {figures}")


if __name__ == "__main__":
    main()
