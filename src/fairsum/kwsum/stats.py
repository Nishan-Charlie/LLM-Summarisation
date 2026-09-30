"""Paired tests, corrections and summary tables for the registered hypotheses.

Every comparison is paired: two conditions scored on the same collection by the
same model. Only scored rows with a non-missing metric enter a pair. Holm
correction runs across the registered tests within each (model, dataset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as st

from fairsum.kwsum.collections import LEVELS, SKEWED_LEVELS
from fairsum.kwsum.compliance import SCORED_STATUSES

PAIR_KEYS = ["model", "dataset", "collection_id", "level"]
_LEVEL_NEG_SHARE = {level: n / (p + n) for level, (p, n) in LEVELS.items()}


def holm(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = order.size
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def wilcoxon_p(diffs: np.ndarray, alternative: str) -> float:
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size < 5:
        return float("nan")
    if np.allclose(diffs, 0):
        return 1.0
    return float(st.wilcoxon(diffs, alternative=alternative, zero_method="wilcox").pvalue)


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def aligned_pairs(scores: pd.DataFrame, metric: str, cond_a: str, cond_b: str,
                  levels: Sequence[str] | None = None, only_if: str | None = None) -> pd.DataFrame:
    """Paired rows (same model, dataset, collection) of two conditions, diff = a - b.

    `only_if` keeps only rows where that boolean column is true (H3 runs on the
    collections where negation actually changed the prompt). Duplicate rows
    raise instead of silently multiplying n.
    """
    s = scores[scores["status"].isin(SCORED_STATUSES) & scores[metric].notna()]
    if levels:
        s = s[s["level"].isin(levels)]
    if only_if:
        s = s[s[only_if].fillna(False).astype(bool)]
    a = s[s["condition"] == cond_a][PAIR_KEYS + [metric]]
    b = s[s["condition"] == cond_b][PAIR_KEYS + [metric]]
    for side, frame_ in ((cond_a, a), (cond_b, b)):
        dup = frame_.duplicated(PAIR_KEYS, keep=False)
        if dup.any():
            raise ValueError(f"{int(dup.sum())} duplicate rows for condition {side!r} "
                             f"(e.g. {frame_[dup].iloc[0][PAIR_KEYS].to_dict()}); de-duplicate the store")
    pairs = a.merge(b, on=PAIR_KEYS, suffixes=("_a", "_b"))
    pairs["diff"] = pairs[f"{metric}_a"].astype(float) - pairs[f"{metric}_b"].astype(float)
    return pairs


def level_interaction_p(pairs: pd.DataFrame, high: Sequence[str] = ("pos88", "neg88"),
                        low: Sequence[str] = ("pos75", "neg75")) -> float:
    """One-sided test that the paired effect is larger at the extreme skew."""
    hi = pairs.loc[pairs["level"].isin(high), "diff"].to_numpy(dtype=float)
    lo = pairs.loc[pairs["level"].isin(low), "diff"].to_numpy(dtype=float)
    if hi.size < 5 or lo.size < 5:
        return float("nan")
    return float(st.mannwhitneyu(hi, lo, alternative="greater").pvalue)


def interaction_effect(pairs: pd.DataFrame, high: Sequence[str] = ("pos88", "neg88"),
                       low: Sequence[str] = ("pos75", "neg75"), n_boot: int = 5000,
                       seed: int = 0) -> tuple[float, float, float]:
    """Mean paired effect at the extreme skew minus at the moderate skew, with a
    bootstrap 95% CI (each group resampled independently). H1c's effect size."""
    hi = pairs.loc[pairs["level"].isin(high), "diff"].to_numpy(dtype=float)
    lo = pairs.loc[pairs["level"].isin(low), "diff"].to_numpy(dtype=float)
    if hi.size == 0 or lo.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = (rng.choice(hi, size=(n_boot, hi.size)).mean(axis=1)
            - rng.choice(lo, size=(n_boot, lo.size)).mean(axis=1))
    return float(hi.mean() - lo.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


@dataclass(frozen=True)
class HypothesisTest:
    hypothesis: str
    cond_a: str
    cond_b: str
    metric: str
    alternative: str  # "less", "greater" or "noninferior" (diff = a - b)
    levels: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    margin: float | None = None
    kind: str = "paired"  # or "interaction"
    only_if: str | None = None  # boolean score column a row must have true


SK = SKEWED_LEVELS
REGISTERED_TESTS: tuple[HypothesisTest, ...] = (
    HypothesisTest("H1a", "kw_global", "direct", "unsupported_rate", "less", SK),
    HypothesisTest("H1b", "kw_global", "direct", "majority_amplification", "greater", SK),
    HypothesisTest("H1c", "kw_global", "direct", "majority_amplification", "greater", SK, kind="interaction"),
    HypothesisTest("H2a", "kw_strat", "kw_global", "majority_amplification", "less", SK),
    HypothesisTest("H2b", "kw_strat", "kw_global", "coverage_gap", "less", SK),
    HypothesisTest("H2c", "kw_strat", "kw_global", "unsupported_rate", "noninferior", margin=0.02),
    # Only where negation changed the prompt: elsewhere the two prompts are identical.
    HypothesisTest("H3", "kw_strat_neg", "kw_strat", "contradiction_rate", "less",
                   only_if="negation_changed"),
    HypothesisTest("H4a", "key_refer", "refer", "unsupported_rate", "less"),
    HypothesisTest("H4b", "key_refer", "refer", "count_abs_error", "less"),
    HypothesisTest("H4c", "key_refer", "refer", "delta_p", "noninferior", margin=0.05),
    HypothesisTest("H4d", "kw_strat_neg", "direct", "delta_p", "less", models=("llama3.2-3b",)),
)


def run_tests(scores: pd.DataFrame,
              tests: Sequence[HypothesisTest] = REGISTERED_TESTS) -> pd.DataFrame:
    rows = []
    for (model, dataset), sub in scores.groupby(["model", "dataset"]):
        for t in tests:
            if t.models and model not in t.models:
                continue
            if t.metric not in sub.columns:
                continue
            if t.only_if and t.only_if not in sub.columns:
                continue
            pairs = aligned_pairs(sub, t.metric, t.cond_a, t.cond_b, t.levels, t.only_if)
            d = pairs["diff"].to_numpy(dtype=float)
            mean_d = float(d.mean()) if d.size else float("nan")
            median_d = float(np.median(d)) if d.size else float("nan")
            if t.kind == "interaction":
                p = level_interaction_p(pairs)
                mean_d, lo, hi = interaction_effect(pairs)  # effect = extreme - moderate
                median_d = float("nan")
            else:
                p = float("nan") if t.alternative == "noninferior" else wilcoxon_p(d, t.alternative)
                lo, hi = bootstrap_ci(d)
            rows.append({
                "model": model, "dataset": dataset, "hypothesis": t.hypothesis,
                "comparison": f"{t.cond_a} vs {t.cond_b}", "metric": t.metric,
                "alternative": t.alternative, "kind": t.kind, "margin": t.margin,
                "subset": t.only_if or "all", "n": int(d.size),
                "mean_diff": mean_d, "median_diff": median_d,
                "ci_lo": lo, "ci_hi": hi, "p": p,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = out.groupby(["model", "dataset"])["p"].transform(lambda s: holm(s.to_numpy()))
    out["supported"] = out.apply(_supported, axis=1)
    return out


def _supported(row: pd.Series) -> bool:
    if row["alternative"] == "noninferior":
        return bool(row["n"] > 0 and row["ci_hi"] < row["margin"])
    if np.isnan(row["p_holm"]) or row["p_holm"] >= 0.05:
        return False
    if row["kind"] == "interaction":
        return True
    return bool(row["mean_diff"] < 0) if row["alternative"] == "less" else bool(row["mean_diff"] > 0)


def per_level_effects(scores: pd.DataFrame,
                      tests: Sequence[HypothesisTest] = REGISTERED_TESTS) -> pd.DataFrame:
    """Descriptive per-level paired effects for the level-restricted tests.

    Uncorrected and not part of the registered family: the registered H1a/H1b
    pool the skewed levels (spec §2). This table shows where an effect sits.
    """
    rows = []
    for (model, dataset), sub in scores.groupby(["model", "dataset"]):
        for t in tests:
            if t.kind != "paired" or not t.levels or t.alternative == "noninferior":
                continue
            if t.metric not in sub.columns:
                continue
            for level in t.levels:
                pairs = aligned_pairs(sub, t.metric, t.cond_a, t.cond_b, [level], t.only_if)
                d = pairs["diff"].to_numpy(dtype=float)
                lo, hi = bootstrap_ci(d)
                rows.append({
                    "model": model, "dataset": dataset, "hypothesis": t.hypothesis,
                    "comparison": f"{t.cond_a} vs {t.cond_b}", "metric": t.metric, "level": level,
                    "n": int(d.size), "mean_diff": float(d.mean()) if d.size else float("nan"),
                    "ci_lo": lo, "ci_hi": hi, "p_uncorrected": wilcoxon_p(d, t.alternative),
                })
    return pd.DataFrame(rows)


def polarity_only_view(scores: pd.DataFrame) -> pd.DataFrame:
    """Scores with the proportion metrics taken from the polarity model alone.

    Running the registered tests on this view is the robustness check for the
    neutral gate's asymmetry (it neutralises positive text more than negative).
    """
    out = scores.copy()
    for metric in ("summary_neg_share", "delta_p", "majority_amplification"):
        alt = f"{metric}_polarity_only"
        if alt in out.columns:
            out[metric] = out[alt]
    return out


def worst_case_fill(scores: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity analysis (spec §5): score every failure as badly as possible.

    Unscored rows get unsupported and contradiction rates of 1 and the largest
    |dp| their level allows. Their status becomes `worst_case`, so the default
    tables still exclude them unless asked.
    """
    out = scores.copy()
    bad = ~out["status"].isin(SCORED_STATUSES)
    neg_share = out["level"].map(_LEVEL_NEG_SHARE)
    out.loc[bad, "unsupported_rate"] = 1.0
    out.loc[bad, "contradiction_rate"] = 1.0
    out.loc[bad, "delta_p"] = np.maximum(neg_share, 1 - neg_share)[bad]
    out.loc[bad, "status"] = "worst_case"
    return out


def condition_means(scores: pd.DataFrame, metrics: Sequence[str],
                    by: Sequence[str] = ("model", "dataset", "condition"),
                    statuses: frozenset[str] = SCORED_STATUSES) -> pd.DataFrame:
    s = scores[scores["status"].isin(statuses)]
    cols = [m for m in metrics if m in s.columns]
    grouped = s.groupby(list(by))
    means = grouped[cols].mean(numeric_only=True)
    means["n_scored"] = grouped.size()
    return means.reset_index()


def compliance_table(scores: pd.DataFrame) -> pd.DataFrame:
    counts = scores.groupby(["model", "condition", "status"]).size().unstack(fill_value=0)
    total = counts.sum(axis=1)
    scored = counts[[c for c in counts.columns if c in SCORED_STATUSES]].sum(axis=1)
    out = counts.copy()
    out["total"] = total
    out["scored_share"] = scored / total
    return out.reset_index()
