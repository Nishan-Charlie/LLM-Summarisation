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
    ma_from_misclassification,
    per_level_effects,
    within_collection_effects,
    deattributed_view,
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


def test_the_deattributed_view_swaps_in_the_attribution_robust_nli_metrics():
    f = pd.DataFrame({"unsupported_rate": [0.8], "unsupported_rate_deattr": [0.5],
                      "contradiction_rate": [0.1], "contradiction_rate_deattr": [0.2],
                      "coverage_gap": [0.3], "coverage_gap_deattr": [0.1],
                      "delta_p": [0.4]})
    r = deattributed_view(f)
    assert r.loc[0, "unsupported_rate"] == 0.5 and r.loc[0, "contradiction_rate"] == 0.2
    assert r.loc[0, "coverage_gap"] == 0.1
    assert r.loc[0, "delta_p"] == 0.4  # proportion metrics do not use NLI


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


def test_misclassification_alone_pulls_majority_amplification_below_zero():
    # Polar statements misclassified 7% of the time, in either direction: a summary that
    # matches a 7:1 collection exactly is measured as under-representing the majority.
    assert ma_from_misclassification(0.875, 0.07) == pytest.approx(-0.0525)
    assert ma_from_misclassification(0.75, 0.07) == pytest.approx(-0.035)
    assert ma_from_misclassification(0.5, 0.2) == pytest.approx(0.0)


def length_frame(seed=0):
    """Coverage rises 0.002 per word; kw_a adds 0.05 beyond its extra 50 words, kw_b adds nothing."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(60):
        base = rng.normal(0.3, 0.05)
        for cond, extra_words, effect in (("direct", 0, 0.0), ("kw_a", 50, 0.05), ("kw_b", 50, 0.0)):
            words = 100 + extra_words + rng.normal(0, 15)
            rows.append({"collection_id": f"c{c}", "condition": cond, "n_words": words,
                         "recall_minority": base + effect + 0.002 * words + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_within_collection_effects_separate_length_from_the_prompt():
    df = length_frame()
    raw = within_collection_effects(df, "recall_minority", ["kw_a", "kw_b"], "direct")
    adj = within_collection_effects(df, "recall_minority", ["kw_a", "kw_b"], "direct", covariate="n_words")
    assert raw["kw_a"][0] == pytest.approx(0.15, abs=0.01)   # 0.05 + 50 words x 0.002
    assert raw["kw_b"][0] == pytest.approx(0.10, abs=0.01)   # length alone
    assert adj["kw_a"][0] == pytest.approx(0.05, abs=0.01)
    assert adj["kw_b"][0] == pytest.approx(0.0, abs=0.01)
    est, lo, hi = adj["kw_a"]
    assert lo < est < hi and lo > 0                           # interval from whole-collection resampling
    assert adj == within_collection_effects(df, "recall_minority", ["kw_a", "kw_b"], "direct",
                                            covariate="n_words")  # deterministic


# --- Amendment 9: entity-clustered inference and equivalence labels -------------------------

from fairsum.kwsum.stats import cluster_bootstrap_ci, run_tests_clustered, tost_label  # noqa: E402


def test_cluster_bootstrap_widens_the_interval_when_values_repeat_within_clusters():
    rng = np.random.default_rng(0)
    cluster_values = rng.normal(size=10)
    values = np.repeat(cluster_values, 10)
    clusters = np.repeat([f"e{i}" for i in range(10)], 10)
    naive_lo, naive_hi = bootstrap_ci(values, seed=1)
    lo, hi = cluster_bootstrap_ci(values, clusters, seed=1)
    assert (lo, hi) == cluster_bootstrap_ci(values, clusters, seed=1)
    assert lo < values.mean() < hi
    assert (hi - lo) > 2 * (naive_hi - naive_lo)


def test_cluster_bootstrap_with_singleton_clusters_is_the_ordinary_bootstrap():
    d = np.random.default_rng(3).normal(size=60)
    lo, hi = cluster_bootstrap_ci(d, np.arange(60).astype(str), seed=2)
    naive_lo, naive_hi = bootstrap_ci(d, seed=2)
    assert abs((hi - lo) - (naive_hi - naive_lo)) < 0.1 * (naive_hi - naive_lo)


def test_clustered_tests_count_entities_and_test_entity_means():
    f = frame()
    entity_of = {f"c{i}": f"e{i // 4}" for i in range(40)}  # 10 entities, 4 collections each
    res = run_tests_clustered(f, entity_of, tests=[t for t in REGISTERED_TESTS if t.hypothesis == "H1a"])
    row = res.iloc[0]
    assert row["n"] == 40 and row["n_entities"] == 10
    assert bool(row["supported"]) and row["mean_diff"] < 0


def test_clustered_noninferiority_reports_an_equivalence_label():
    f = frame()
    f["kw_strat_shift"] = 0.0
    rows = f[f["condition"] == "kw_global"].copy()
    rows["condition"] = "kw_strat"
    g = pd.concat([f, rows], ignore_index=True)
    entity_of = {f"c{i}": f"e{i}" for i in range(40)}
    res = run_tests_clustered(g, entity_of, tests=[t for t in REGISTERED_TESTS if t.hypothesis == "H2c"])
    row = res.iloc[0]
    assert bool(row["supported"]) and row["tost"] == "equivalent"


@pytest.mark.parametrize("lo,hi,label", [
    (-0.01, 0.01, "equivalent"),
    (-0.05, 0.01, "non-inferior"),
    (-0.01, 0.03, "inconclusive"),
    (0.025, 0.04, "inferior"),
])
def test_tost_labels_follow_the_margin(lo, hi, label):
    assert tost_label(lo, hi, 0.02) == label


def test_the_control_tests_are_those_registered_in_amendments_9_and_10():
    from fairsum.kwsum.stats import CONTROL_TESTS
    got = {(t.hypothesis, t.cond_a, t.cond_b, t.metric, t.alternative, t.levels) for t in CONTROL_TESTS}
    from fairsum.kwsum.collections import SKEWED_LEVELS
    assert got == {
        ("C1", "kw_strat", "direct_len", "recall_minority", "greater", tuple(SKEWED_LEVELS)),
        ("C2", "counts_neutral", "refer_oracle", "majority_amplification", "greater", tuple(SKEWED_LEVELS)),
        ("C3", "counts_reflect", "refer_oracle", "majority_amplification", "greater", tuple(SKEWED_LEVELS)),
    }


def test_clustered_tests_can_correct_across_cells_of_one_hypothesis():
    f = pd.concat([frame().assign(dataset=d) for d in ("d1", "d2")], ignore_index=True)
    f["collection_id"] = f["dataset"] + f["collection_id"]
    entity_of = {cid: cid for cid in f["collection_id"].unique()}
    tests = [t for t in REGISTERED_TESTS if t.hypothesis in ("H1a", "H1b")]
    by_cell = run_tests_clustered(f, entity_of, tests=tests)
    by_hyp = run_tests_clustered(f, entity_of, tests=tests, holm_by=("hypothesis",))
    for h, g in by_hyp.groupby("hypothesis"):
        assert np.allclose(g["p_holm"], holm(g["p"].to_numpy()), equal_nan=True)
    assert not np.allclose(by_cell["p_holm"], by_hyp["p_holm"], rtol=1e-6, atol=0)


def test_within_collection_effects_can_resample_entities():
    df = length_frame()
    plain = within_collection_effects(df, "recall_minority", ["kw_a"], "direct", covariate="n_words")
    identity = {f"c{c}": f"c{c}" for c in range(60)}
    assert within_collection_effects(df, "recall_minority", ["kw_a"], "direct", covariate="n_words",
                                     clusters=identity) == plain
    six = {f"c{c}": f"e{c % 6}" for c in range(60)}       # 6 entities of 10 collections each
    est, lo, hi = within_collection_effects(df, "recall_minority", ["kw_a"], "direct",
                                            covariate="n_words", clusters=six)["kw_a"]
    assert est == pytest.approx(plain["kw_a"][0]) and lo < est < hi
