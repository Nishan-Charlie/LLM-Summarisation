"""Paired tests and corrections for the registered hypotheses (spec §2)."""

import numpy as np
import pandas as pd
import pytest

from fairsum.kwsum.compliance import SCORED_STATUSES
from fairsum.kwsum.stats import (
    REGISTERED_TESTS,
    aligned_pairs,
    bootstrap_ci,
    compliance_table,
    condition_means,
    holm,
    level_interaction_p,
    per_level_effects,
    polarity_only_view,
    run_tests,
    wilcoxon_p,
    worst_case_fill,
)


def test_holm_known_values():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_leaves_nan_alone():
    out = holm([0.01, np.nan])
    assert out[0] == pytest.approx(0.01) and np.isnan(out[1])


def test_wilcoxon_small_n_is_nan():
    assert np.isnan(wilcoxon_p(np.array([0.1, 0.2]), "less"))


def test_wilcoxon_all_zero_is_one():
    assert wilcoxon_p(np.zeros(10), "less") == 1.0


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    d = np.random.default_rng(0).normal(size=50)
    lo, hi = bootstrap_ci(d, seed=1)
    assert (lo, hi) == bootstrap_ci(d, seed=1)
    assert lo < d.mean() < hi


def frame():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        level = ["pos75", "pos88", "neg75", "neg88"][i % 4]
        for condition, shift in (("direct", 0.0), ("kw_global", -0.2)):
            rows.append({
                "model": "m", "dataset": "d", "collection_id": f"c{i}", "level": level,
                "condition": condition, "status": "ok",
                "unsupported_rate": 0.5 + shift + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def test_aligned_pairs_match_on_collection():
    pairs = aligned_pairs(frame(), "unsupported_rate", "kw_global", "direct")
    assert len(pairs) == 40 and (pairs["diff"] < 0).all()


def test_unscored_rows_are_excluded_from_pairs():
    f = frame()
    f.loc[0, "status"] = "empty"
    assert len(aligned_pairs(f, "unsupported_rate", "kw_global", "direct")) == 39


def test_run_tests_supports_a_clear_effect():
    res = run_tests(frame(), tests=[t for t in REGISTERED_TESTS if t.hypothesis == "H1a"])
    row = res.iloc[0]
    assert bool(row["supported"]) and row["p_holm"] < 0.05 and row["mean_diff"] < 0


def test_level_interaction_detects_a_bigger_effect_at_the_extreme():
    pairs = pd.DataFrame({
        "level": ["pos88"] * 20 + ["pos75"] * 20,
        "diff": list(np.linspace(0.3, 0.5, 20)) + list(np.linspace(0.0, 0.1, 20)),
    })
    assert level_interaction_p(pairs) < 0.01


def test_compliance_table_counts_scored_share():
    f = frame()
    f.loc[0, "status"] = "no_summary"
    table = compliance_table(f)
    direct = table[(table["model"] == "m") & (table["condition"] == "direct")].iloc[0]
    assert direct["scored_share"] == pytest.approx(39 / 40)


def test_worst_case_fill_assigns_maximal_values_to_failures():
    f = frame()
    f["delta_p"] = 0.1
    f["contradiction_rate"] = 0.0
    f.loc[0, "status"] = "no_summary"  # row 0 is level pos75: source neg share 0.25
    w = worst_case_fill(f)
    assert w.loc[0, "unsupported_rate"] == 1.0 and w.loc[0, "contradiction_rate"] == 1.0
    assert w.loc[0, "delta_p"] == pytest.approx(0.75)
    assert w.loc[0, "status"] == "worst_case"
    assert w.loc[1, "delta_p"] == pytest.approx(0.1)  # scored rows untouched
    means = condition_means(w, ["unsupported_rate"], statuses=SCORED_STATUSES | {"worst_case"})
    direct = means[means["condition"] == "direct"].iloc[0]
    assert direct["n_scored"] == 40


def test_the_robustness_view_swaps_in_the_polarity_only_metrics():
    f = pd.DataFrame({"delta_p": [0.1], "delta_p_polarity_only": [0.3],
                      "majority_amplification": [0.0], "majority_amplification_polarity_only": [-0.2]})
    r = polarity_only_view(f)
    assert r.loc[0, "delta_p"] == 0.3 and r.loc[0, "majority_amplification"] == -0.2


def test_duplicate_rows_are_rejected_not_multiplied():
    f = pd.concat([frame(), frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        aligned_pairs(f, "unsupported_rate", "kw_global", "direct")


def test_a_test_can_be_restricted_to_manipulated_collections():
    f = frame()
    f["negation_changed"] = f["collection_id"].isin([f"c{i}" for i in range(10)])
    pairs = aligned_pairs(f, "unsupported_rate", "kw_global", "direct", only_if="negation_changed")
    assert len(pairs) == 10


def test_h3_is_registered_on_negation_changed_collections_only():
    (h3,) = [t for t in REGISTERED_TESTS if t.hypothesis == "H3"]
    assert h3.only_if == "negation_changed"


def test_the_interaction_reports_the_difference_in_effects_between_skews():
    rows = []
    for i in range(40):
        level = ["pos75", "pos88"][i % 2]
        shift = 0.30 if level == "pos88" else 0.10
        for condition, s in (("direct", 0.0), ("kw_global", shift)):
            rows.append({"model": "m", "dataset": "d", "collection_id": f"c{i}", "level": level,
                         "condition": condition, "status": "ok", "majority_amplification": s})
    res = run_tests(pd.DataFrame(rows), tests=[t for t in REGISTERED_TESTS if t.hypothesis == "H1c"])
    assert res.iloc[0]["mean_diff"] == pytest.approx(0.20)  # 0.30 at 7:1 minus 0.10 at 6:2
    assert res.iloc[0]["ci_lo"] - 1e-9 <= 0.20 <= res.iloc[0]["ci_hi"] + 1e-9


def test_per_level_effects_are_descriptive_rows_per_level():
    table = per_level_effects(frame(), tests=[t for t in REGISTERED_TESTS if t.hypothesis == "H1a"])
    assert set(table["level"]) == {"pos75", "pos88", "neg75", "neg88"}
    assert (table["n"] == 10).all() and (table["mean_diff"] < 0).all()
