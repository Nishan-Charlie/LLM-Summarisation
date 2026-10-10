"""Every result of the keyword study in one long CSV, plus a wide table of condition means.

    python scripts/kw_export_results.py --config configs/kw_week1.toml

Writes results/kw/week1/tables/all_results.csv, one number per row:

    section    condition_mean | level_mean | hypothesis_test | hypothesis_by_level |
               length_control | stated_counts | compliance | misclassification_bound
    model, dataset (amazon, drugs, or both), level, condition, comparison, metric,
    view       registered | polarity_only | attribution_robust (which instrument a number uses)
    value, ci_lo, ci_hi, n, p, p_holm, supported, note

and results/kw/week1/tables/all_condition_means_wide.csv (one row per model x dataset x
condition, one column per metric). Means use the current scorer version and scored summaries
only; their intervals are normal approximations (mean +/- 1.96 SE). The hypothesis tests and
the length control keep the paper's bootstrap intervals and Holm-corrected p-values.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

from fairsum.kwsum.compliance import SCORED_STATUSES
from fairsum.kwsum.config import load_config
from fairsum.kwsum.propositions import split_summary
from fairsum.kwsum.scoring import SCORER_VERSION
from fairsum.kwsum.stats import ma_from_misclassification, within_collection_effects
from fairsum.kwsum.store import read_jsonl

SKEWED = ["pos75", "pos88", "neg75", "neg88"]
TRUE_MAJORITY_SHARE = {"pos75": 0.75, "pos88": 0.875, "neg75": 0.75, "neg88": 0.875}

# metric -> (instrument view, what it measures)
METRICS = {
    "unsupported_rate": ("registered", "share of summary statements entailed by no single review (NLI, threshold 0.5)"),
    "unsupported_rate_deattr": ("attribution_robust", "unsupported rate with reviewer frames removed ('reviewers found X' -> 'X' / 'I found X')"),
    "support_score": ("registered", "mean over statements of the highest entailment probability against any review"),
    "support_score_deattr": ("attribution_robust", "support score with reviewer frames removed"),
    "contradiction_rate": ("registered", "share of statements contradicted by a review and supported by none"),
    "contradiction_rate_deattr": ("attribution_robust", "contradiction rate with reviewer frames removed"),
    "delta_p": ("registered", "|summary negative share - source negative share|, hybrid sentiment instrument"),
    "delta_p_polarity_only": ("polarity_only", "|summary negative share - source negative share|, SiEBERT alone"),
    "majority_amplification": ("registered", "summary majority share - source majority share (skewed levels; < 0 = majority under-represented), hybrid instrument"),
    "majority_amplification_polarity_only": ("polarity_only", "majority amplification with SiEBERT alone"),
    "summary_neg_share": ("registered", "negative share of the summary's polar statements, hybrid instrument"),
    "summary_neg_share_polarity_only": ("polarity_only", "negative share of the summary's polar statements, SiEBERT alone"),
    "source_neg_share": ("registered", "negative share of the 8 source reviews (by rating)"),
    "recall_minority": ("registered", "share of minority reviews that support at least one statement (skewed levels)"),
    "recall_minority_deattr": ("attribution_robust", "minority coverage with reviewer frames removed"),
    "recall_majority": ("registered", "share of majority reviews that support at least one statement (skewed levels)"),
    "recall_majority_deattr": ("attribution_robust", "majority coverage with reviewer frames removed"),
    "coverage_gap": ("registered", "majority coverage - minority coverage (skewed levels)"),
    "coverage_gap_deattr": ("attribution_robust", "coverage gap with reviewer frames removed"),
    "recall_pos": ("registered", "share of positive reviews that support at least one statement"),
    "recall_neg": ("registered", "share of negative reviews that support at least one statement"),
    "attribution_share_minority": ("registered", "share of supported statements supported by a minority review (skewed levels)"),
    "keyword_uptake": ("registered", "share of the prompt's keyphrases that appear in the summary (keyword prompts)"),
    "keyword_uptake_minority": ("registered", "share of the minority group's keyphrases that appear in the summary (grouped prompts, skewed levels)"),
    "n_words": ("registered", "summary length in words"),
    "n_props": ("registered", "statements in the summary after splitting"),
    "n_polar_props": ("registered", "statements the hybrid instrument calls positive or negative"),
    "n_deattributed": ("attribution_robust", "statements with a reviewer frame that the robust score rewrote"),
    "n_count_statements": ("registered", "statements that state review counts"),
    "n_meta": ("registered", "statements about the task rather than the product (dropped before scoring)"),
    "count_statements_parsed": ("registered", "review counts parsed from the summary"),
    "count_abs_error": ("registered", "mean absolute error of stated review counts (summaries that state counts)"),
    "count_wrong": ("registered", "1 if any stated count is wrong (summaries that state counts)"),
    "invented_numbers": ("registered", "numbers in the statements that appear in no review"),
    "bur": ("registered", "BUR (Zhang et al. 2023): 1 if either opinion's summary share is below 0.8 x its source share"),
    "uer": ("registered", "UER (Zhang et al. 2023): mean under-representation of the two opinions"),
    "sof": ("registered", "SOF (Zhang et al. 2023): mean absolute deviation of the shortfalls"),
    "spd": ("registered", "summed absolute difference between source and summary opinion distributions (= 2 x UER)"),
}
VIEW_OF_FILE = {"hypotheses.csv": "registered", "hypotheses_robustness_polarity_only.csv": "polarity_only",
                "hypotheses_robustness_deattributed.csv": "attribution_robust"}
COLUMNS = ["section", "model", "dataset", "level", "condition", "comparison", "metric", "view", "value",
           "ci_lo", "ci_hi", "n", "p", "p_holm", "supported", "note"]


def summarise(values: pd.Series) -> tuple[float, float, float, int]:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan
    return m, m - 1.96 * se, m + 1.96 * se, int(v.size)


def mean_rows(s: pd.DataFrame, section: str, by: list[str]) -> list[dict]:
    rows = []
    for keys, g in s.groupby(by):
        ids = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        for metric, (view, note) in METRICS.items():
            if metric not in g.columns:
                continue
            m, lo, hi, n = summarise(g[metric])
            if n:
                rows.append({"section": section, **ids, "metric": metric, "view": view, "value": m,
                             "ci_lo": lo, "ci_hi": hi, "n": n, "note": note})
    return rows


def stated_split_rows(cfg, models: list[str]) -> list[dict]:
    """REFER prompts: how often stated splits on skewed inputs are 4:4, and how often they give
    the true majority too small a share (as in the paper)."""
    pairs: dict[tuple[str, str], list] = {}
    for g in read_jsonl(cfg.generations_path):
        if g["model"] not in models or g["condition"] not in ("refer", "cot_refer") or g["level"] not in SKEWED:
            continue
        counts = split_summary(g["raw"], drop_unterminated_tail=g["status"] == "truncated").counts
        pos = next((n for n, p in counts if p == "positive" and 0 <= n <= 8), None)
        neg = next((n for n, p in counts if p == "negative" and 0 <= n <= 8), None)
        if pos is None or neg is None or pos + neg == 0:
            continue
        majority = pos if g["level"].startswith("pos") else neg
        pairs.setdefault((g["model"], g["condition"]), []).append(
            (pos, neg, majority / (pos + neg) < TRUE_MAJORITY_SHARE[g["level"]]))
    rows = []
    for (m, c), ps in sorted(pairs.items()):
        n = len(ps)
        (mode, freq), = Counter((p, q) for p, q, _ in ps).most_common(1)
        base = {"section": "stated_counts", "model": m, "dataset": "both", "level": "skewed", "condition": c,
                "view": "registered", "n": n}
        rows += [{**base, "metric": "stated_split_even_share", "value": sum(p == q for p, q, _ in ps) / n,
                  "note": "share of stated splits on skewed inputs that are an even 4:4"},
                 {**base, "metric": "stated_split_understates_majority_share", "value": sum(u for *_, u in ps) / n,
                  "note": "share of stated splits giving the true majority a smaller share than it has"},
                 {**base, "metric": "stated_split_mode_share", "value": freq / n,
                  "note": f"share of the most common stated split ({mode[0]}:{mode[1]})"}]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tables = cfg.results_dir / "tables"

    s = pd.DataFrame(read_jsonl(cfg.scores_path))
    s = s[(s["scorer_version"] == SCORER_VERSION) & (s["split"] == "test")]
    models = sorted(s["model"].unique())
    scored = s[s["status"].isin(SCORED_STATUSES)].copy()
    both = scored.assign(dataset="both")
    rows: list[dict] = []

    # 1-2. Condition means (per dataset and pooled) and per-level means, every metric.
    rows += mean_rows(pd.concat([scored, both]), "condition_mean", ["model", "dataset", "condition"])
    rows += mean_rows(scored, "level_mean", ["model", "dataset", "level", "condition"])

    # 3-4. Registered tests in all three instrument views, and per-level descriptive effects.
    for fname, view in VIEW_OF_FILE.items():
        h = pd.read_csv(tables / fname)
        for r in h.itertuples(index=False):
            rows.append({"section": "hypothesis_test", "model": r.model, "dataset": r.dataset,
                         "condition": r.hypothesis, "comparison": r.comparison, "metric": r.metric, "view": view,
                         "value": r.mean_diff, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "n": r.n, "p": r.p,
                         "p_holm": r.p_holm, "supported": bool(r.supported),
                         "note": f"{r.kind}; alternative {r.alternative}"
                                 + (f"; margin {r.margin}" if pd.notna(r.margin) else "")
                                 + (f"; subset {r.subset}" if isinstance(r.subset, str) and r.subset else "")})
    lv = pd.read_csv(tables / "hypotheses_by_level_descriptive.csv")
    for r in lv.itertuples(index=False):
        rows.append({"section": "hypothesis_by_level", "model": r.model, "dataset": r.dataset, "level": r.level,
                     "condition": r.hypothesis, "comparison": r.comparison, "metric": r.metric, "view": "registered",
                     "value": r.mean_diff, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "n": r.n, "p": r.p_uncorrected,
                     "note": "descriptive per-level paired difference; p not corrected"})

    # 5. Coverage gains as measured and at equal summary length.
    skew = scored[scored["level"].isin(SKEWED)]
    for m in models:
        d = skew[skew["model"] == m]
        n_coll = d["collection_id"].nunique()
        for covariate, label in ((None, "as_measured"), ("n_words", "equal_length")):
            eff = within_collection_effects(d, "recall_minority", ["kw_global", "kw_strat"], "direct", covariate=covariate)
            for c, (est, lo, hi) in eff.items():
                rows.append({"section": "length_control", "model": m, "dataset": "both", "level": "skewed",
                             "condition": c, "comparison": f"{c} vs direct", "metric": f"recall_minority_gain_{label}",
                             "view": "registered", "value": est, "ci_lo": lo, "ci_hi": hi, "n": n_coll,
                             "note": "within-collection regression; n = collections; interval resamples collections"})

    # 6. Stated counts: how often prompts state counts, and how often they are exact.
    for (m, c), g in scored.groupby(["model", "condition"]):
        claims = g[g["count_abs_error"].notna()]
        rows.append({"section": "stated_counts", "model": m, "dataset": "both", "level": "all", "condition": c,
                     "metric": "states_counts_share", "view": "registered", "value": len(claims) / len(g),
                     "n": len(g), "note": "share of scored summaries that state review counts"})
        for level, cl in [*claims.groupby("level"), ("skewed", claims[claims["level"].isin(SKEWED)])]:
            if len(cl):
                rows.append({"section": "stated_counts", "model": m, "dataset": "both", "level": level, "condition": c,
                             "metric": "counts_exact_share", "view": "registered",
                             "value": float((cl["count_wrong"] == 0).mean()), "n": len(cl),
                             "note": "share of count-stating summaries whose counts are all exact"})
    rows += stated_split_rows(cfg, models)

    # 7. Output compliance.
    comp = pd.read_csv(tables / "compliance.csv")
    for r in comp.itertuples(index=False):
        for col in ("ok", "truncated", "refusal", "no_summary"):
            rows.append({"section": "compliance", "model": r.model, "dataset": "both", "level": "all",
                         "condition": r.condition, "metric": f"{col}_share", "view": "registered",
                         "value": getattr(r, col) / r.total, "n": r.total, "note": "share of the 600 prompts"})

    # 8. What misclassification alone does to majority amplification.
    hybrid = json.loads((cfg.results_dir / "sentiment_validation_hybrid.json").read_text(encoding="utf-8"))
    error = 1 - hybrid["polar_accuracy"]
    for level, share in TRUE_MAJORITY_SHARE.items():
        rows.append({"section": "misclassification_bound", "level": level, "metric": "majority_amplification_artefact",
                     "view": "registered", "value": ma_from_misclassification(share, error),
                     "note": f"MA a perfectly proportional summary would show with {error:.1%} symmetric polar error"})

    out = pd.DataFrame(rows).reindex(columns=COLUMNS)
    out.to_csv(tables / "all_results.csv", index=False)
    wide = (out[out["section"] == "condition_mean"]
            .pivot_table(index=["model", "dataset", "condition"], columns="metric", values="value").reset_index())
    wide.to_csv(tables / "all_condition_means_wide.csv", index=False)
    print(f"{len(out)} rows -> {tables / 'all_results.csv'}")
    print(out.groupby("section").size().to_string())
    print(f"{len(wide)} rows x {wide.shape[1]} columns -> {tables / 'all_condition_means_wide.csv'}")


if __name__ == "__main__":
    main()
