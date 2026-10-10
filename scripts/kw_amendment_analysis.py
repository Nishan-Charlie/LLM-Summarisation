"""Amendment 9 analyses (docs/superpowers/specs/2026-10-02-kw-amendment-9.md).

    python scripts/kw_amendment_analysis.py --config configs/kw_week1.toml --controls configs/kw_controls.toml

Writes beside the registered week-1 tables, which it never touches:
  hypotheses_clustered.csv (+ _polarity_only, _deattributed)  registered tests, entity as the unit
  ma_by_direction.csv                                         MA by majority side, entity-clustered CIs
and into the controls run's tables, for every model whose controls are fully scored:
  controls_tests.csv (+ _polarity_only, _deattributed)        C1 and C2
  controls_descriptive.csv                                    length match; counts_neutral vs direct
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from fairsum.kwsum.collections import LEVELS, SKEWED_LEVELS, read_collections
from fairsum.kwsum.compliance import SCORED_STATUSES
from fairsum.kwsum.config import COLLECTIONS_PATH, load_config
from fairsum.kwsum.scoring import SCORER_VERSION
from fairsum.kwsum.stats import (
    CONTROL_TESTS, aligned_pairs, cluster_bootstrap_ci, deattributed_view, ma_from_misclassification,
    polarity_only_view, run_tests_clustered,
)
from fairsum.kwsum.store import read_jsonl

VIEWS = {"": lambda s: s, "_polarity_only": polarity_only_view, "_deattributed": deattributed_view}
DIRECTION = {"pos75": "positive", "pos88": "positive", "neg75": "negative", "neg88": "negative"}


def load_scores(cfg) -> pd.DataFrame:
    s = pd.DataFrame(read_jsonl(cfg.scores_path))
    if s.empty:
        return s
    return s[(s["scorer_version"].astype(str) == SCORER_VERSION) & (s["split"] == "test")].copy()


def majority_share(level: str) -> float:
    p, n = LEVELS[level]
    return max(p, n) / (p + n)


def ma_by_direction(scores: pd.DataFrame, entity_of: dict, errors: dict) -> pd.DataFrame:
    """Mean majority amplification per model, dataset (and both), prompt and majority side,
    with entity-resampled intervals and the envelope that classifier error alone produces."""
    s = scores[scores["status"].isin(SCORED_STATUSES) & scores["level"].isin(SKEWED_LEVELS)].copy()
    s["direction"] = s["level"].map(DIRECTION)
    s["entity"] = s["collection_id"].map(entity_of)
    rows = []
    for view, col in (("registered", "majority_amplification"),
                      ("polarity_only", "majority_amplification_polarity_only")):
        v = s.dropna(subset=[col])
        for dataset in ("amazon", "drugs", "both"):
            d = v if dataset == "both" else v[v["dataset"] == dataset]
            for (model, cond, direction), g in d.groupby(["model", "condition", "direction"]):
                lo, hi = cluster_bootstrap_ci(g[col].to_numpy(float), g["entity"].to_numpy())
                env = float(np.mean([ma_from_misclassification(majority_share(lv), errors[view])
                                     for lv in g["level"]]))
                rows.append({"view": view, "model": model, "dataset": dataset, "condition": cond,
                             "direction": direction, "n": len(g), "mean": float(g[col].mean()),
                             "ci_lo": lo, "ci_hi": hi, "envelope": env,
                             "beyond_envelope": bool(hi < env)})
    return pd.DataFrame(rows)


def abs_distortion(week1: pd.DataFrame, entity_of: dict) -> pd.DataFrame:
    """Amendment 10, descriptive: the size of the proportion distortion regardless of its sign, as the
    proportion check on the primary question (H2a, registered as 'majority amplification falls', is
    reported as a mechanistic prediction). Negative differences mean less distortion."""
    s = week1.copy()
    s["abs_majority_gap"] = s["majority_amplification"].abs()
    rows = []
    for dataset in ("amazon", "drugs", "both"):
        d = s if dataset == "both" else s[s["dataset"] == dataset]
        for model, sub in d.groupby("model"):
            for a, b in (("kw_strat", "kw_global"), ("kw_strat", "direct"), ("kw_global", "direct")):
                for metric, levels in (("abs_majority_gap", SKEWED_LEVELS), ("delta_p", None)):
                    pairs = aligned_pairs(sub, metric, a, b, levels)
                    if pairs.empty:
                        continue
                    lo, hi = cluster_bootstrap_ci(pairs["diff"].to_numpy(float),
                                                  pairs["collection_id"].map(entity_of).to_numpy())
                    rows.append({"model": model, "dataset": dataset, "comparison": f"{a} vs {b}",
                                 "metric": metric, "n": len(pairs), "mean_diff": float(pairs["diff"].mean()),
                                 "ci_lo": lo, "ci_hi": hi, "worse": bool(lo > 0), "better": bool(hi < 0)})
    return pd.DataFrame(rows)


def control_descriptives(both: pd.DataFrame, entity_of: dict) -> pd.DataFrame:
    """Paired descriptive contrasts for the controls (not tests): did the length match hold,
    and what do the true counts do without the balance instruction."""
    specs = [
        ("length_match", "direct_len", "kw_strat", "n_words", None),
        ("length_match", "direct_len", "direct", "n_words", None),
        ("coverage", "direct_len", "direct", "recall_minority", SKEWED_LEVELS),
        ("coverage", "kw_strat", "direct", "recall_minority", SKEWED_LEVELS),
        ("counts", "counts_neutral", "direct", "majority_amplification", SKEWED_LEVELS),
        ("counts", "counts_neutral", "direct", "delta_p", None),
        ("counts", "refer_oracle", "direct", "majority_amplification", SKEWED_LEVELS),
        ("counts", "counts_neutral", "refer_oracle", "delta_p", None),
        ("counts", "counts_neutral", "refer_oracle", "unsupported_rate", None),
        ("counts", "counts_neutral", "refer_oracle", "recall_minority", SKEWED_LEVELS),
        # amendment 10: the word "balanced" alone
        ("counts", "counts_reflect", "refer_oracle", "majority_amplification", SKEWED_LEVELS),
        ("counts", "counts_reflect", "counts_neutral", "majority_amplification", SKEWED_LEVELS),
        ("counts", "counts_reflect", "direct", "majority_amplification", SKEWED_LEVELS),
    ]
    rows = []
    for dataset in ("amazon", "drugs", "both"):
        d = both if dataset == "both" else both[both["dataset"] == dataset]
        for model, sub in d.groupby("model"):
            for family, a, b, metric, levels in specs:
                if metric not in sub.columns:
                    continue
                pairs = aligned_pairs(sub, metric, a, b, levels)
                if pairs.empty:
                    continue
                lo, hi = cluster_bootstrap_ci(pairs["diff"].to_numpy(float),
                                              pairs["collection_id"].map(entity_of).to_numpy())
                rows.append({"family": family, "model": model, "dataset": dataset,
                             "comparison": f"{a} vs {b}", "metric": metric, "n": len(pairs),
                             "mean_a": float(pairs[f"{metric}_a"].mean()),
                             "mean_b": float(pairs[f"{metric}_b"].mean()),
                             "mean_diff": float(pairs["diff"].mean()), "ci_lo": lo, "ci_hi": hi})
        # Count statements: how often each prompt states counts, and how often they are right.
        for (model, cond), g in d[d["condition"].isin(["counts_neutral", "counts_reflect", "refer_oracle", "direct"])].groupby(
                ["model", "condition"]):
            g = g[g["status"].isin(SCORED_STATUSES)]
            states = g["n_count_statements"].fillna(0) > 0
            wrong = g.loc[states, "count_wrong"] if "count_wrong" in g.columns else pd.Series(dtype=float)
            rows.append({"family": "count_statements", "model": model, "dataset": dataset,
                         "comparison": cond, "metric": "states_counts_share", "n": len(g),
                         "mean_a": float(states.mean()) if len(g) else float("nan")})
            if len(wrong.dropna()):
                rows.append({"family": "count_statements", "model": model, "dataset": dataset,
                             "comparison": cond, "metric": "count_wrong_share", "n": int(wrong.notna().sum()),
                             "mean_a": float(wrong.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--controls", default="configs/kw_controls.toml")
    args = ap.parse_args()
    cfg, ccfg = load_config(args.config), load_config(args.controls)
    entity_of = {c.collection_id: f"{c.dataset}:{c.entity}" for c in read_collections(COLLECTIONS_PATH)}
    week1 = load_scores(cfg)
    tables = cfg.results_dir / "tables"

    for suffix, view in VIEWS.items():
        run_tests_clustered(view(week1), entity_of).to_csv(tables / f"hypotheses_clustered{suffix}.csv", index=False)
    siebert = json.loads((cfg.results_dir / "sentiment_validation.json").read_text(encoding="utf-8"))
    hybrid = json.loads((cfg.results_dir / "sentiment_validation_hybrid.json").read_text(encoding="utf-8"))
    errors = {"registered": 1 - hybrid["polar_accuracy"], "polarity_only": 1 - siebert["accuracy"]}

    abs_distortion(week1, entity_of).to_csv(tables / "abs_distortion.csv", index=False)

    # Each control prompt counts for a model once all its test summaries are scored, so C1 and C2
    # (amendment 9) do not wait for counts_reflect (amendment 10), and C3 runs as soon as it can.
    controls = load_scores(ccfg)
    n_sets = len(week1["collection_id"].unique())
    done_by: dict[str, list[str]] = {}
    if not controls.empty:
        counts = controls.groupby(["condition", "model"]).size()
        for cond in ccfg.conditions:
            done_by[cond] = sorted(m for (c, m), n in counts.items() if c == cond and n >= n_sets)
    complete = sorted(set(done_by.get("direct_len", [])) & set(done_by.get("counts_neutral", [])))
    reflect = done_by.get("counts_reflect", [])
    ready = controls.iloc[0:0] if controls.empty else controls[
        (controls["model"].isin(complete) & controls["condition"].isin(["direct_len", "counts_neutral"]))
        | (controls["model"].isin(reflect) & (controls["condition"] == "counts_reflect"))]
    ma_by_direction(pd.concat([week1, ready], ignore_index=True), entity_of, errors).to_csv(
        tables / "ma_by_direction.csv", index=False)
    print(f"clustered tests, absolute distortion and MA by direction written to {tables}")

    ctables = ccfg.results_dir / "tables"
    ctables.mkdir(parents=True, exist_ok=True)
    (ctables / "controls_status.json").write_text(json.dumps(
        {"complete_models": complete, "complete_by_condition": done_by, "expected_per_condition": n_sets,
         "rows_per_model": controls.groupby("model").size().to_dict() if not controls.empty else {}},
        indent=2), encoding="utf-8")
    if not complete:
        print("no model has fully scored controls yet")
        return
    base = week1[week1["model"].isin(complete) & week1["condition"].isin(["direct", "kw_strat", "refer_oracle"])]
    both = pd.concat([base, ready], ignore_index=True)
    amend9 = [t for t in CONTROL_TESTS if t.hypothesis in ("C1", "C2")]
    c3 = [t for t in CONTROL_TESTS if t.hypothesis == "C3"]
    for suffix, view in VIEWS.items():
        parts = [run_tests_clustered(view(both), entity_of, tests=amend9, holm_by=("hypothesis",))]
        if reflect:
            parts.append(run_tests_clustered(view(both[both["model"].isin(reflect)]), entity_of, tests=c3,
                                             holm_by=("hypothesis",)))
        pd.concat(parts, ignore_index=True).to_csv(ctables / f"controls_tests{suffix}.csv", index=False)
    control_descriptives(both, entity_of).to_csv(ctables / "controls_descriptive.csv", index=False)
    print(f"controls C1/C2 for {complete}, C3 for {reflect or 'no model yet'}, written to {ctables}")


if __name__ == "__main__":
    main()
