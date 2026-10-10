"""Numbers, tables and figures for the paper, generated from the results (none typed by hand).

    python scripts/kw_paper_assets.py --config configs/kw_week1.toml
    python scripts/kw_paper_assets.py --config configs/kw_week1.toml --scorer-version 3

Writes paper/generated/numbers.tex (read in the text as \\res{key}), paper/generated/tab_*.tex
and paper/figures/*.pdf. A model enters the tables only once all of its test summaries are
scored by the chosen scorer version; partly scored models are listed as pending.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter

import numpy as np
import pandas as pd

from fairsum.kwsum import figures as F  # noqa: F401  (sets the shared matplotlib style)
from fairsum.kwsum.collections import read_collections
from fairsum.kwsum.compliance import SCORED_STATUSES
from fairsum.kwsum.config import COLLECTIONS_PATH, KEYWORDS_PATH, MANIFEST_PATH, REPO_ROOT, load_config
from fairsum.kwsum.human_check import GUIDE
from fairsum.kwsum.keywords import minority_share
from fairsum.kwsum.paper import (fmt, fmt_ci, fmt_p, latex_table, macro_file, panel_table, sign_cells,
                                 supported_cells)
from fairsum.kwsum.propositions import split_summary
from fairsum.kwsum.scoring import SCORER_VERSION
from fairsum.kwsum.stats import (cluster_bootstrap_ci, deattributed_view, holm, ma_from_misclassification, polarity_only_view,
                                 run_tests, within_collection_effects)
from fairsum.kwsum.store import read_jsonl

import matplotlib.pyplot as plt  # noqa: E402  (after the style import)

PAPER = REPO_ROOT / "paper"
GEN, FIG = PAPER / "generated", PAPER / "figures"

MODELS = {"llama3.2-1b": ("l1b", "Llama-3.2-1B"), "llama3.2-3b": ("l3b", "Llama-3.2-3B"),
          "llama3.1-8b": ("l8b", "Llama-3.1-8B"), "qwen2.5-7b": ("q7b", "Qwen2.5-7B"),
          "mistral-7b": ("m7b", "Mistral-7B")}
DATASETS = {"amazon": "Amazon", "drugs": "Drugs.com"}
CONDITIONS = ["direct", "refer", "cot_refer", "refer_oracle", "kw_global", "kw_strat",
              "kw_strat_neg", "key_refer", "key_oracle"]
LEVELS = {"bal": "4:4", "pos75": "6:2", "pos88": "7:1", "neg75": "2:6", "neg88": "1:7"}
SKEWED = ["pos75", "pos88", "neg75", "neg88"]
# column -> (macro slug, computed on the skewed levels only)
METRICS = {
    "unsupported_rate": ("unsup", False), "unsupported_rate_deattr": ("unsupd", False),
    "contradiction_rate": ("contra", False), "contradiction_rate_deattr": ("contrad", False),
    "delta_p": ("dp", False), "delta_p_polarity_only": ("dppol", False),
    "majority_amplification": ("ma", True), "majority_amplification_polarity_only": ("mapol", True),
    "recall_minority": ("recmin", True), "recall_minority_deattr": ("recmind", True),
    "coverage_gap": ("gap", True), "coverage_gap_deattr": ("gapd", True),
    "n_words": ("words", False),
    "keyword_uptake": ("uptake", False), "keyword_uptake_minority": ("uptakemin", True),
}
NLI_METRICS = {"unsupported_rate", "contradiction_rate", "coverage_gap"}
PROPORTION_METRICS = {"delta_p", "majority_amplification"}
HYPOTHESES = ["H1a", "H1b", "H1c", "H2a", "H2b", "H2c", "H3", "H4a", "H4b", "H4c", "H4d"]


def cslug(condition: str) -> str:
    return condition.replace("_", "")


# Reader-facing names of the prompts, used in every table and figure (the code names stay in the data).
NAMES = {"direct": "Plain", "refer": "REFER", "cot_refer": "REFER-CoT", "refer_oracle": "REFER-Counts",
         "kw_global": "Keywords-All", "kw_strat": "Keywords-Grouped", "kw_strat_neg": "Keywords-Grouped-Neg",
         "key_refer": "REFER+Keywords", "key_oracle": "REFER-Counts+Keywords",
         "direct_len": "Plain-Length", "counts_neutral": "Counts-Only", "counts_reflect": "Counts-Reflect"}


def pname(condition: str) -> str:
    return NAMES.get(condition, condition)


def code(condition: str) -> str:
    return r"\textsf{" + pname(condition) + "}"


def load_scores(cfg, version: str) -> pd.DataFrame:
    s = pd.DataFrame(read_jsonl(cfg.scores_path))
    s = s[(s["scorer_version"] == version) & (s["split"] == "test")].copy()
    return s


def data_numbers(cfg, V: dict) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for ds, pool in manifest["pools"].items():
        V[f"{ds}-reviews"] = f"{pool['reviews']:,}"
        V[f"{ds}-entities"] = f"{pool['entities']:,}"
        V[f"{ds}-pos"] = fmt(pool["positive_share"], 1, percent=True)
        V[f"{ds}-eligible-neg88"] = f"{pool['eligible_entities_by_level']['neg88']:,}"
    V["n-per-cell"] = str(manifest["n_test"])
    V["n-test-collections"] = f"{manifest['n_test'] * len(manifest['levels']) * len(manifest['pools']):,}"
    V["n-conditions"] = str(len(cfg.conditions))
    V["n-per-model"] = f"{manifest['n_test'] * len(manifest['levels']) * len(manifest['pools']) * len(cfg.conditions):,}"

    rows = []
    kws = {r["collection_id"]: r for r in read_jsonl(KEYWORDS_PATH)}
    for c in read_collections(COLLECTIONS_PATH):
        if c.split != "test":
            continue
        share = minority_share(kws[c.collection_id]["global"], c)
        if share is not None:
            rows.append({"dataset": c.dataset, "level": c.level, "kw": share})
    # Entities can recur across levels (amendment 9 makes the entity the unit of inference).
    test = [c for c in read_collections(COLLECTIONS_PATH) if c.split == "test"]
    for ds in DATASETS:
        per = Counter(c.entity for c in test if c.dataset == ds)
        V[f"entities-test-{ds}"] = str(len(per))
        V[f"entities-repeat-{ds}"] = str(sum(n > 1 for n in per.values()))
    km = pd.DataFrame(rows).groupby(["dataset", "level"])["kw"].mean()
    for (ds, level), v in km.items():
        V[f"kwmin-{ds}-{level}"] = fmt(v)
    review_share = {"pos75": 0.25, "neg75": 0.25, "pos88": 0.125, "neg88": 0.125}
    kw_table = pd.DataFrame([{"Dataset": DATASETS[ds], "Level": LEVELS[level],
                              "Minority share of reviews": fmt(review_share[level]),
                              "Minority share of keywords": fmt(v)}
                             for (ds, level), v in km.items()])
    (GEN / "tab_kwmin.tex").write_text(latex_table(
        kw_table, caption="Does picking keywords from all reviews leave the minority out? Share of the "
        "\\textsf{Keywords-All} phrases that occur in at least one minority review.", label="tab:kwmin",
        align="llrr"), encoding="utf-8")

    table = pd.DataFrame([{
        "Source": "Amazon Reviews 2023, Health \\& Personal Care" if ds == "amazon" else "Drugs.com (UCI 462)",
        "Usable reviews": f"{p['reviews']:,}", "Entities": f"{p['entities']:,}",
        "Positive": fmt(p["positive_share"], 1, percent=True),
        "Entities with 1:7": f"{p['eligible_entities_by_level']['neg88']:,}",
    } for ds, p in manifest["pools"].items()])
    (GEN / "tab_data.tex").write_text(latex_table(
        table, caption="Review pools after filtering (30--120 words, polar ratings only).",
        label="tab:data", align="p{4.2cm}rrrr", colsep="4pt",
        note="Entities with 1:7 = products or drugs with at least one positive and seven negative "
             "usable reviews, the scarcest level."), encoding="utf-8")


def instrument_numbers(cfg, V: dict) -> None:
    res = cfg.results_dir
    siebert = json.loads((res / "sentiment_validation.json").read_text(encoding="utf-8"))
    hybrid = json.loads((res / "sentiment_validation_hybrid.json").read_text(encoding="utf-8"))
    cardiff_path = res.parent / "archive" / "sentiment_validation_cardiffnlp.json"
    cardiff = json.loads(cardiff_path.read_text(encoding="utf-8"))
    V["sent-n"] = str(siebert["n"])
    V["sent-cardiff-acc"] = fmt(cardiff["accuracy"], 2)
    V["sent-cardiff-drugs"] = fmt(cardiff["by_dataset"]["drugs"], 2)
    V["sent-siebert-acc"] = fmt(siebert["accuracy"], 2)
    V["sent-hybrid-polar"] = fmt(hybrid["polar_accuracy"], 2)
    V["sent-hybrid-neutral-pos"] = fmt(hybrid["neutral_rate_by_polarity"]["positive"], 1, percent=True)
    V["sent-hybrid-neutral-neg"] = fmt(hybrid["neutral_rate_by_polarity"]["negative"], 1, percent=True)
    # How far misclassification alone pulls majority amplification below zero, if summary
    # statements are misclassified as often as reviews are (the human check measures this).
    error = 1 - hybrid["polar_accuracy"]
    V["maart-error"] = fmt(error, 0, percent=True)
    V["maart-pos88"] = fmt(ma_from_misclassification(0.875, error))
    V["maart-pos75"] = fmt(ma_from_misclassification(0.75, error))
    V["maart-mean"] = fmt((ma_from_misclassification(0.875, error) + ma_from_misclassification(0.75, error)) / 2)
    table = pd.DataFrame([
        {"Instrument": "cardiffnlp Twitter-RoBERTa (3 classes)", "Accuracy": fmt(cardiff["accuracy"], 2),
         "Amazon": fmt(cardiff["by_dataset"]["amazon"], 2), "Drugs.com": fmt(cardiff["by_dataset"]["drugs"], 2),
         "Gate ($\\geq$ 0.90)": "fails"},
        {"Instrument": "SiEBERT RoBERTa-large (2 classes)", "Accuracy": fmt(siebert["accuracy"], 2),
         "Amazon": fmt(siebert["by_dataset"]["amazon"], 2), "Drugs.com": fmt(siebert["by_dataset"]["drugs"], 2),
         "Gate ($\\geq$ 0.90)": "passes"},
        {"Instrument": "Hybrid (used): neutral gate + SiEBERT polarity",
         "Accuracy": fmt(hybrid["polar_accuracy"], 2) + "$^{a}$",
         "Amazon": fmt(hybrid["by_dataset"]["amazon"], 2) + "$^{b}$",
         "Drugs.com": fmt(hybrid["by_dataset"]["drugs"], 2) + "$^{b}$", "Gate ($\\geq$ 0.90)": "passes"},
    ])
    note = (f"{siebert['n']} source reviews sampled at random, labelled by star rating. "
            f"$^{{a}}$Accuracy on the reviews the hybrid calls positive or negative. "
            f"$^{{b}}$A neutral call counted as an error. The neutral gate labels "
            f"{V['sent-hybrid-neutral-pos']} of positive and {V['sent-hybrid-neutral-neg']} of negative "
            "reviews neutral, so every proportion result is repeated with the polarity model alone.")
    (GEN / "tab_instruments.tex").write_text(latex_table(
        table, caption="Validation of the sentiment instruments against star ratings.",
        label="tab:instruments", align="p{4.6cm}rrrl", note=note, colsep="4pt"), encoding="utf-8")


def model_numbers(s: pd.DataFrame, m: str, V: dict) -> None:
    slug = MODELS[m][0]
    ms = s[s["model"] == m]
    scored = ms[ms["status"].isin(SCORED_STATUSES)]
    V[f"scored-{slug}"] = fmt(len(scored) / len(ms), 1, percent=True)
    V[f"refusals-{slug}"] = str(int((ms["status"] == "refusal").sum()))
    skew = scored[scored["level"].isin(SKEWED)]
    for c in CONDITIONS:
        mc, sc = ms[ms["condition"] == c], scored[scored["condition"] == c]
        V[f"trunc-{slug}-{cslug(c)}"] = fmt((mc["status"] == "truncated").mean(), 1, percent=True)
        for col, (mslug, skew_only) in METRICS.items():
            src = skew[skew["condition"] == c] if skew_only else sc
            if col in src.columns:
                V[f"mean-{mslug}-{slug}-{cslug(c)}"] = fmt(src[col].mean(), 1 if col == "n_words" else 3)
        if "n_deattributed" in sc.columns and sc["n_deattributed"].notna().any():
            V[f"deattrshare-{slug}-{cslug(c)}"] = fmt(sc["n_deattributed"].sum() / sc["n_props"].sum(), 1,
                                                       percent=True)
        claims = sc[sc["count_abs_error"].notna()]
        V[f"countrate-{slug}-{cslug(c)}"] = fmt(len(claims) / max(len(sc), 1), 1, percent=True)
        for level in LEVELS:
            cl = claims[claims["level"] == level]
            V[f"countn-{slug}-{cslug(c)}-{level}"] = str(len(cl))
            V[f"countexact-{slug}-{cslug(c)}-{level}"] = (fmt((cl["count_wrong"] == 0).mean(), 0, percent=True)
                                                          if len(cl) else "--")
        sk = claims[claims["level"].isin(SKEWED)]
        V[f"countexact-{slug}-{cslug(c)}-skewed"] = fmt((sk["count_wrong"] == 0).mean(), 0, percent=True) \
            if len(sk) else "--"
    bal = scored[(scored["condition"] == "direct") & (scored["level"] == "bal")]
    V[f"negbal-{slug}"] = fmt(bal["summary_neg_share"].mean())
    if "summary_neg_share_polarity_only" in bal.columns:
        V[f"negbalpol-{slug}"] = fmt(bal["summary_neg_share_polarity_only"].mean())


TRUE_MAJORITY_SHARE = {"pos75": 0.75, "pos88": 0.875, "neg75": 0.75, "neg88": 0.875}


def stated_split_numbers(cfg, m: str, V: dict) -> None:
    """The positive:negative split REFER outputs state on skewed inputs: how often it is 4:4, and
    how often it gives the true majority a smaller share than it has (toward balance or past it)."""
    slug = MODELS[m][0]
    for c in ("refer", "cot_refer"):
        pairs, understated = [], 0
        for g in read_jsonl(cfg.generations_path):
            if g["model"] != m or g["condition"] != c or g["level"] not in SKEWED:
                continue
            counts = split_summary(g["raw"], drop_unterminated_tail=g["status"] == "truncated").counts
            pos = next((n for n, p in counts if p == "positive" and 0 <= n <= 8), None)
            neg = next((n for n, p in counts if p == "negative" and 0 <= n <= 8), None)
            if pos is None or neg is None or pos + neg == 0:
                continue
            pairs.append((pos, neg))
            majority = pos if g["level"].startswith("pos") else neg
            understated += majority / (pos + neg) < TRUE_MAJORITY_SHARE[g["level"]]
        if not pairs:
            continue
        (mode, freq), = Counter(pairs).most_common(1)
        V[f"statedmode-{slug}-{cslug(c)}"] = f"{mode[0]}:{mode[1]}"
        V[f"statedmodeshare-{slug}-{cslug(c)}"] = fmt(freq / len(pairs), 0, percent=True)
        V[f"statedn-{slug}-{cslug(c)}"] = str(len(pairs))
        V[f"statedeven-{slug}-{cslug(c)}"] = fmt(sum(p == n for p, n in pairs) / len(pairs), 0, percent=True)
        V[f"statedunder-{slug}-{cslug(c)}"] = fmt(understated / len(pairs), 0, percent=True)


def means_table(m: str, V: dict) -> str:
    slug, name = MODELS[m]
    has_d = f"mean-unsupd-{slug}-direct" in V and V[f"mean-unsupd-{slug}-direct"] != "--"
    rows = []
    for c in CONDITIONS:
        g = lambda k: V.get(f"mean-{k}-{slug}-{cslug(c)}", "--")  # noqa: E731
        row = {"Prompt": code(c), "Unsup.": g("unsup")}
        if has_d:
            row["Unsup.\\ corr."] = g("unsupd")
        row.update({"Contr.": g("contra"), "Prop.\\ err.": g("dp"), "Maj.\\ gap": g("ma"),
                    "Min.\\ cov.": g("recmin"), "Words": g("words")})
        rows.append(row)
    df = pd.DataFrame(rows)
    note = ("Means over both review sources. Unsup.: share of summary statements that no single review "
            "supports" + ("; corr.: the same with the corrected check (Section~\\ref{sec:faithfulness})"
                          if has_d else "") +
            ". Contr.: statements that a review contradicts and none supports. Prop.\\ err.: gap between "
            "the share of negative opinion in the summary and in the reviews (all sets). Maj.\\ gap: the "
            "majority's share of the summary minus its share of the reviews (one-sided sets; below zero, the "
            "summary looks more balanced than the reviews). Min.\\ cov.: share of minority reviews whose "
            "content appears in the summary (one-sided sets).")
    return latex_table(df, caption=f"Average results by prompt for {name}.", label=f"tab:means-{slug}",
                       align="l" + "r" * (len(df.columns) - 1), note=note, colsep="3pt", size=r"\scriptsize")


def hypothesis_numbers(views: dict[str, pd.DataFrame], V: dict) -> None:
    for view, tests in views.items():
        for r in tests.itertuples(index=False):
            if r.model not in MODELS:
                continue
            key = f"h-{view}-{r.hypothesis.lower()}-{MODELS[r.model][0]}-{r.dataset}"
            V[key + "-diff"] = fmt(r.mean_diff, signed=True)
            V[key + "-ci"] = fmt_ci(r.mean_diff, r.ci_lo, r.ci_hi)
            V[key + "-p"] = fmt_p(r.p_holm)
            V[key + "-n"] = str(int(r.n))
            V[key + "-sup"] = "yes" if r.supported else "no"


def pooled_numbers(s: pd.DataFrame, views: dict[str, pd.DataFrame], V: dict) -> None:
    """Cross-model statements: in how many model x dataset x prompt cells the majority is
    under-represented, in how many model x dataset cells each test is supported, and the
    range of the share of statements the attribution rewrite changes."""
    skew = s[s["status"].isin(SCORED_STATUSES) & s["level"].isin(SKEWED)]
    for col, key in (("majority_amplification", "ma"), ("majority_amplification_polarity_only", "mapol")):
        if col in skew.columns:
            negative, cells, largest = sign_cells(skew, ["model", "dataset", "condition"], col)
            V[f"{key}-negcells"], V[f"{key}-cells"], V[f"{key}-max"] = str(negative), str(cells), fmt(largest)
    for view, tests in views.items():
        for h in HYPOTHESES:
            sup, cells = supported_cells(tests[tests["model"].isin(MODELS)], h)
            V[f"hsup-{view}-{h.lower()}"], V[f"hcells-{view}-{h.lower()}"] = str(sup), str(cells)
    scored = s[s["status"].isin(SCORED_STATUSES)]
    if "n_deattributed" in scored.columns and scored["n_deattributed"].notna().any():
        g = scored.groupby(["model", "condition"])[["n_deattributed", "n_props"]].sum()
        share = g["n_deattributed"] / g["n_props"]
        V["deattrshare-min"], V["deattrshare-max"] = fmt(share.min(), 0, percent=True), fmt(share.max(), 0,
                                                                                           percent=True)


def _supported_in(tests: pd.DataFrame, m: str, h: str) -> list[str]:
    t = tests[(tests["model"] == m) & (tests["hypothesis"] == h)]
    return [ds for ds in DATASETS if bool(t[t["dataset"] == ds]["supported"].any())]


def hypothesis_matrix(views: dict[str, pd.DataFrame], models: list[str]) -> str:
    reg = views["reg"]
    metric_of = reg.drop_duplicates("hypothesis").set_index("hypothesis")["metric"].to_dict()
    rows = []
    for h in HYPOTHESES:
        if h not in metric_of:
            continue
        metric = metric_of[h]
        robust = "deattr" if metric in NLI_METRICS else "pol" if metric in PROPORTION_METRICS else None
        row = {"Test": h}
        for m in models:
            if reg[(reg["model"] == m) & (reg["hypothesis"] == h)].empty:
                row[MODELS[m][1]] = "n/a"
                continue
            got = _supported_in(reg, m, h)
            cell = ", ".join("A" if d == "amazon" else "D" for d in got) or "--"
            if robust and robust in views and _supported_in(views[robust], m, h) != got:
                cell += "$^{\\dagger}$"
            row[MODELS[m][1]] = cell
        rows.append(row)
    note = ("A: the prediction holds on Amazon; D: on Drugs.com; --: on neither (paired tests corrected for the "
            "number of tests; for ``does not rise by more than'' predictions, the 95\\% interval). $^{\\dagger}$: "
            "the verdict changes with the alternative measure (SiEBERT alone for proportions; the "
            "attribution-robust score for faithfulness and coverage). n/a: planned for another model only.")
    return latex_table(pd.DataFrame(rows), caption="The planned tests of Table~\\ref{tab:hypotheses}: where each "
                       "prediction holds.",
                       label="tab:hyp-matrix", align="l" + "c" * len(models), note=note, colsep="4pt")


SHORT_METRIC = {"unsupported_rate": "unsupported", "majority_amplification": "majority gap",
                "coverage_gap": "coverage gap", "contradiction_rate": "contradicted",
                "count_abs_error": "count error", "delta_p": "prop.\\ error"}


def hypothesis_full(views: dict[str, pd.DataFrame], m: str) -> str:
    reg = views["reg"]
    rows = []
    for h in HYPOTHESES:
        t = reg[(reg["model"] == m) & (reg["hypothesis"] == h)]
        if t.empty:
            continue
        row = {"Test": h, "Metric": SHORT_METRIC.get(t["metric"].iloc[0], t["metric"].iloc[0])}
        for ds, label in DATASETS.items():
            r = t[t["dataset"] == ds]
            row[f"{label} $\\Delta$ [95\\% CI]"] = fmt_ci(r["mean_diff"].iloc[0], r["ci_lo"].iloc[0],
                                                        r["ci_hi"].iloc[0]) if len(r) else "--"
            row[f"$p_{{\\mathrm{{{label[0]}}}}}$"] = fmt_p(r["p_holm"].iloc[0]) if len(r) else "--"
        rows.append(row)
    return latex_table(pd.DataFrame(rows), caption=f"Planned tests for {MODELS[m][1]}: average difference "
                       "between the two prompts of Table~\\ref{tab:hypotheses} (first minus second) on the same "
                       "review sets, with its 95\\% interval, and the $p$-value corrected for the number of tests.",
                       label=f"tab:hyp-{MODELS[m][0]}", align="llrrrr", size=r"\scriptsize",
                       colsep="3pt")


ROBUST_VIEWS = {"pol": ("the alternative sentiment classifier (SiEBERT alone)", ["H1b", "H1c", "H2a", "H4c", "H4d"]),
                "deattr": ("the attribution-robust faithfulness score", ["H1a", "H2b", "H2c", "H3", "H4a"])}


def robustness_table(views: dict[str, pd.DataFrame], view: str, models: list[str]) -> str:
    what, hyps = ROBUST_VIEWS[view]
    tests = views[view]
    rows = []
    for h in hyps:
        for ds, label in DATASETS.items():
            row = {"Test": h, "Data": label}
            for m in models:
                r = tests[(tests["model"] == m) & (tests["hypothesis"] == h) & (tests["dataset"] == ds)]
                row[MODELS[m][1]] = "n/a" if r.empty else (
                    fmt(r["mean_diff"].iloc[0], signed=True) + ("$^{*}$" if bool(r["supported"].iloc[0]) else ""))
            rows.append(row)
    return latex_table(pd.DataFrame(rows), caption=f"The planned tests repeated with {what}: average paired "
                       "difference for the tests that depend on it ($^{*}$: the prediction holds, by the same rule).", label=f"tab:robust-{view}", align="ll" + "r" * len(models),
                       size=r"\footnotesize", colsep="2.5pt")


LENGTH_CONDITIONS = ["kw_global", "kw_strat"]


def length_numbers(s: pd.DataFrame, models: list[str], V: dict, controls: pd.DataFrame | None = None) -> str:
    """Minority coverage gain over the plain prompt, as measured and holding summary length
    constant (within-collection regression, skewed levels); keyword prompts write longer
    summaries, and longer summaries can cover more reviews. With the controls scored, the
    length-matched plain prompt joins the regression, so length varies by design as well."""
    skew = s[s["status"].isin(SCORED_STATUSES) & s["level"].isin(SKEWED)]
    ctrl = (controls[controls["status"].isin(SCORED_STATUSES) & controls["level"].isin(SKEWED)
                     & (controls["condition"] == "direct_len")] if controls is not None and len(controls) else None)
    ent = entity_map()
    rows, positive = [], Counter()
    for m in models:
        slug, name = MODELS[m]
        d = skew[skew["model"] == m]
        raw = within_collection_effects(d, "recall_minority", LENGTH_CONDITIONS, "direct", clusters=ent)
        adj = within_collection_effects(d, "recall_minority", LENGTH_CONDITIONS, "direct", covariate="n_words",
                                        clusters=ent)
        with_ctrl = None
        if ctrl is not None and (ctrl["model"] == m).any():
            dd = pd.concat([d[d["condition"].isin(["direct", "kw_strat"])], ctrl[ctrl["model"] == m]])
            with_ctrl = within_collection_effects(dd, "recall_minority", ["kw_strat", "direct_len"], "direct",
                                                  covariate="n_words", clusters=ent)
            V[f"lenctl-{slug}"] = fmt_ci(*with_ctrl["kw_strat"])
            positive["ctl"] += with_ctrl["kw_strat"][1] > 0
        for c in LENGTH_CONDITIONS:
            V[f"lenraw-{slug}-{cslug(c)}"] = fmt_ci(*raw[c])
            V[f"lenadj-{slug}-{cslug(c)}"] = fmt_ci(*adj[c])
            positive[c] += adj[c][1] > 0
            row = {"Model": name, "Prompt": code(c), "As measured": fmt_ci(*raw[c]),
                   "At equal length": fmt_ci(*adj[c])}
            if ctrl is not None:
                row["With control"] = (
                    fmt_ci(*with_ctrl["kw_strat"]) if with_ctrl and c == "kw_strat" else "")
            rows.append(row)
    for c in LENGTH_CONDITIONS:
        V[f"lenpos-{cslug(c)}"] = str(positive[c])
    V["lenpos-ctl"] = str(positive["ctl"])
    df = pd.DataFrame(rows)
    return latex_table(df, caption="Gain in minority coverage over \\textsf{Plain} on one-sided sets, as measured "
                       "and at equal summary length (a regression within each review set, with summary length as "
                       "a covariate). With control: \\textsf{Plain-Length} is added to the regression, so that "
                       "length also varies by design. 95\\% intervals resample products and drugs.", label="tab:length",
                       align="ll" + "r" * (len(df.columns) - 2), size=r"\scriptsize", colsep="2pt")


def appendix_tables(models: list[str], views: dict[str, pd.DataFrame]) -> str:
    """The appendix block for whichever models are complete, so adding a model needs no edit."""
    lines = [r"\input{generated/tab_hyp_matrix}", r"\input{generated/tab_controls}", r"\input{generated/tab_counts}"]
    lines += [rf"\input{{generated/tab_robust_{v}}}" for v in ROBUST_VIEWS if v in views]
    lines.append(r"\input{generated/tab_cluster}")
    lines.append(r"\input{generated/tab_length}")
    for m in models:
        slug = MODELS[m][0]
        lines += [rf"\input{{generated/tab_means_{slug}}}", rf"\input{{generated/tab_hyp_{slug}}}"]
    return "\n".join(lines + [r"\input{generated/tab_compliance}"]) + "\n"


# Two tables rather than one, so that each fits on a page with five models.
CROSS_TABLES = {
    "fair": ("Minority coverage and opinion proportions, by prompt and model (averages over both review sources).",
             [("recmin", "Minority coverage (higher is better)"),
              ("ma", "Majority gap (0 is ideal)"),
              ("dp", "Proportion error (lower is better)")],
             "Minority coverage: share of minority reviews whose content appears in the summary (one-sided sets). "
             "Majority gap: the majority's share of the summary's opinion statements minus its share of the "
             "reviews (one-sided sets; below 0, the summary looks more balanced than the reviews). Proportion "
             "error: gap between the summary's and the reviews' share of negative opinion (all sets). "),
    "faith": ("Unsupported statements, by prompt and model: share of summary statements that no single review "
              "supports (averages over both review sources; lower is better).",
              [("unsup", "Planned check"),
               ("unsupd", "Corrected check")],
             "Corrected check: statements that attribute an opinion to reviewers (``some reviewers say \\ldots'') "
             "are also checked without that phrase (Section~\\ref{sec:faithfulness}). "),
}


def cross_model_table(V: dict, models: list[str], which: str) -> str:
    caption, spec, explain = CROSS_TABLES[which]
    short = {"llama3.2-1b": "Llama 1B", "llama3.2-3b": "Llama 3B", "llama3.1-8b": "Llama 8B",
             "qwen2.5-7b": "Qwen 7B", "mistral-7b": "Mistral 7B"}  # full names in Table models
    panels = []
    for metric, title in spec:
        rows = [{"Prompt": code(c), **{short[m]: V.get(f"mean-{metric}-{MODELS[m][0]}-{cslug(c)}", "--")
                                       for m in models}} for c in CONDITIONS]
        panels.append((title, pd.DataFrame(rows)))
    return panel_table(panels, caption=caption, label=f"tab:cross-{which}", align="l" + "r" * len(models),
                       size=r"\footnotesize", colsep="4pt",
                       note=explain + "Per-model tables with all measures and summary length are in "
                            "\\ref{app:tables}.")


def counts_table(V: dict, models: list[str]) -> str:
    rows = []
    for m in models:
        slug, name = MODELS[m]
        for c in ("refer", "cot_refer", "key_refer"):
            row = {"Model": name, "Prompt": code(c), "States": V[f"countrate-{slug}-{cslug(c)}"]}
            for level, label in LEVELS.items():
                n = V[f"countn-{slug}-{cslug(c)}-{level}"]
                row[f"Exact {label}"] = f"{V[f'countexact-{slug}-{cslug(c)}-{level}']} ({n})"
            rows.append(row)
    return latex_table(pd.DataFrame(rows), caption="Review counts written in the summary: how often prompts that "
                       "ask the model to count actually state the counts, and how often the stated counts are "
                       "right, by split (positive:negative). Number of summaries stating counts in parentheses.",
                       label="tab:counts", align="llr" + "r" * len(LEVELS), size=r"\scriptsize",
                       colsep="3pt")


def compliance_table(V: dict, models: list[str]) -> str:
    rows = []
    for c in CONDITIONS:
        row = {"Prompt": code(c)}
        for m in models:
            slug = MODELS[m][0]
            row[MODELS[m][1]] = f"{V[f'trunc-{slug}-{cslug(c)}']} / {V[f'mean-words-{slug}-{cslug(c)}']}"
        rows.append(row)
    return latex_table(pd.DataFrame(rows), caption="Summary length: share of summaries cut off at the 256-token "
                       "limit / average number of words.", label="tab:compliance", align="l" + "r" * len(models),
                       size=r"\scriptsize", colsep="3pt")


# --- Amendment 9: clustered inference, majority direction, controls --------------------------
# Read from the tables scripts/kw_amendment_analysis.py writes; the registered tables above are
# unchanged, and these are reported beside them.

VIEW_FILES = {"reg": "", "pol": "_polarity_only", "deattr": "_deattributed"}
DIRS = {"positive": "pos", "negative": "neg"}
CONTROL_CONDITIONS = ["direct_len", "counts_neutral", "counts_reflect"]


def entity_map() -> dict[str, str]:
    return {c.collection_id: f"{c.dataset}:{c.entity}" for c in read_collections(COLLECTIONS_PATH)}


def clustered_numbers(cfg, views: dict[str, pd.DataFrame], models: list[str], V: dict) -> str:
    """Registered tests with the entity as the unit, the verdicts that change, and equivalence
    labels for the non-inferiority tests. Returns the comparison table."""
    tables = cfg.results_dir / "tables"
    cl = {v: pd.read_csv(tables / f"hypotheses_clustered{suffix}.csv") for v, suffix in VIEW_FILES.items()}
    changes: dict[str, list[str]] = {}
    for view, t in cl.items():
        t = t[t["model"].isin(models)]
        for h in HYPOTHESES:
            sup, cells = supported_cells(t, h)
            V[f"hsupcl-{view}-{h.lower()}"] = str(sup)
        for r in t.itertuples(index=False):
            key = f"hcl-{view}-{r.hypothesis.lower()}-{MODELS[r.model][0]}-{r.dataset}"
            V[key + "-ci"] = fmt_ci(r.mean_diff, r.ci_lo, r.ci_hi)
            V[key + "-p"] = fmt_p(r.p_holm)
            if isinstance(getattr(r, "tost", None), str):
                V[key + "-tost"] = r.tost
        merged = views[view].merge(t, on=["model", "dataset", "hypothesis"], suffixes=("", "_cl"))
        # A robust view only speaks for the tests whose metric it changes.
        if view == "pol":
            merged = merged[merged["metric"].isin(PROPORTION_METRICS)]
        elif view == "deattr":
            merged = merged[merged["metric"].isin(NLI_METRICS)]
        flips = merged[merged["supported"] != merged["supported_cl"]]
        changes[view] = [f"{MODELS[r.model][1]} on {DATASETS[r.dataset]}, {r.hypothesis} "
                         f"({'supported' if r.supported else 'not supported'} $\\to$ "
                         f"{'supported' if r.supported_cl else 'not supported'}, Holm $p$ "
                         f"{fmt_p(r.p_holm)} $\\to$ {fmt_p(r.p_holm_cl)})" for r in flips.itertuples(index=False)]
        V[f"clchanges-n-{view}"] = str(len(flips))
        V[f"clchanges-{view}"] = "; ".join(changes[view]) or "none"
        for h in ("H2c", "H4c"):
            tt = t[t["hypothesis"] == h]
            for label in ("equivalent", "non-inferior", "inconclusive", "inferior"):
                V[f"tost-{view}-{h.lower()}-{label}"] = str(int((tt["tost"] == label).sum()))
    rows = []
    for h in HYPOTHESES:
        reg = views["reg"]
        if reg[reg["hypothesis"] == h].empty:
            continue
        metric = reg[reg["hypothesis"] == h]["metric"].iloc[0]
        robust = "deattr" if metric in NLI_METRICS else "pol" if metric in PROPORTION_METRICS else None
        cells = supported_cells(reg[reg["model"].isin(models)], h)[1]
        row = {"Test": h, "Cells": str(cells),
               "Coll.": supported_cells(reg[reg["model"].isin(models)], h)[0],
               "Ent.": V[f"hsupcl-reg-{h.lower()}"]}
        if robust:
            rt = views[robust]
            row["Robust coll."] = supported_cells(rt[rt["model"].isin(models)], h)[0]
            row["Robust ent."] = V[f"hsupcl-{robust}-{h.lower()}"]
        else:
            row["Robust coll."] = row["Robust ent."] = "--"
        if h in ("H2c", "H4c"):
            labels = [(lab, V[f"tost-reg-{h.lower()}-{lab}"]) for lab in
                      ("equivalent", "non-inferior", "inconclusive", "inferior")]
            row["Equivalence (TOST)"] = ", ".join(f"{n} {lab}" for lab, n in labels if n != "0")
        else:
            row["Equivalence (TOST)"] = ""
        rows.append(row)
    note = ("Number of the ten model--source combinations in which each planned prediction holds, treating each "
            "review set as independent (as planned) and treating each product or drug as one unit (Ent.), since "
            "some appear in more than one set. Alternative: SiEBERT alone for proportions, the attribution-robust "
            "score for faithfulness and coverage. Equivalence: 90\\% interval against the planned margin. Verdicts "
            "that change when products and drugs are the unit: "
            + (V["clchanges-reg"] if V["clchanges-reg"] != "none" else "none") + " (planned measures); "
            + ("; ".join(changes.get("pol", []) + changes.get("deattr", [])) or "none") + " (alternative measures).")
    return latex_table(pd.DataFrame(rows), caption="Planned tests with review sets and with products or drugs as "
                       "the unit (Coll.: review sets; Ent.: products or drugs).", label="tab:cluster", align="lrrrrrp{3.6cm}", note=note,
                       size=r"\footnotesize", colsep="3pt")


def abs_distortion_numbers(cfg, models: list[str], V: dict) -> None:
    """Amendment 10 (descriptive): in how many model x source cells a prompt makes the proportion
    distortion larger or smaller in size, whatever its sign."""
    path = cfg.results_dir / "tables" / "abs_distortion.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    d = d[d["model"].isin(models)]
    mslug = {"abs_majority_gap": "absgap", "delta_p": "dp"}
    for (cmp_, metric), g in d[d["dataset"] != "both"].groupby(["comparison", "metric"]):
        a, b = cmp_.split(" vs ")
        key = f"absd-{cslug(a)}-{cslug(b)}-{mslug[metric]}"
        V[key + "-worse"], V[key + "-better"] = str(int(g["worse"].sum())), str(int(g["better"].sum()))
        V[key + "-cells"] = str(len(g))
    for r in d[d["dataset"] == "both"].itertuples(index=False):
        a, b = r.comparison.split(" vs ")
        V[f"absd-{cslug(a)}-{cslug(b)}-{mslug[r.metric]}-{MODELS[r.model][0]}"] = fmt_ci(r.mean_diff, r.ci_lo, r.ci_hi)
    both = d[(d["dataset"] == "both") & (d["comparison"] == "kw_strat vs kw_global") & (d["metric"] == "abs_majority_gap")]
    if len(both):
        V["absd-kwstrat-kwglobal-absgap-range"] = f"{fmt(both['mean_diff'].min(), signed=True)} to {fmt(both['mean_diff'].max(), signed=True)}"


def direction_numbers(cfg, models: list[str], V: dict) -> pd.DataFrame:
    """Majority amplification by majority side, against the envelope that classifier error alone
    would produce (amendment 9, descriptive)."""
    d = pd.read_csv(cfg.results_dir / "tables" / "ma_by_direction.csv")
    d = d[d["model"].isin(models)]
    for view, short in (("registered", "reg"), ("polarity_only", "pol")):
        v = d[d["view"] == view]
        V[f"env-{short}"] = fmt(v["envelope"].mean())
        both = v[v["dataset"] == "both"]
        for r in both.itertuples(index=False):
            key = f"madir-{short}-{MODELS[r.model][0]}-{cslug(r.condition)}-{DIRS[r.direction]}"
            V[key] = fmt(r.mean)
            V[key + "-ci"] = fmt_ci(r.mean, r.ci_lo, r.ci_hi)
        for (cond, direction), g in both.groupby(["condition", "direction"]):
            V[f"madirbeyond-{short}-{cslug(cond)}-{DIRS[direction]}"] = str(int(g["beyond_envelope"].sum()))
            V[f"madirrange-{short}-{cslug(cond)}-{DIRS[direction]}"] = (
                f"{fmt(g['mean'].max())} to {fmt(g['mean'].min())}")
        cells = v[v["dataset"] != "both"]
        for (cond, direction), g in cells.groupby(["condition", "direction"]):
            V[f"madirbeyondcells-{short}-{cslug(cond)}-{DIRS[direction]}"] = str(int(g["beyond_envelope"].sum()))
    return d


def control_numbers(ccfg, models: list[str], V: dict) -> str | None:
    """C1 and C2 with their robustness views, the descriptive contrasts, and the condition means
    of the two control prompts. Returns the controls table, or None before they are scored."""
    ctables = ccfg.results_dir / "tables"
    status_path = ctables / "controls_status.json"
    if not status_path.exists():
        V["controls-done"] = "no"
        return None
    done = [m for m in models if m in json.loads(status_path.read_text(encoding="utf-8"))["complete_models"]]
    V["controls-done"] = "yes" if done == models else "partly"
    V["controls-models"] = ", ".join(MODELS[m][1] for m in done) or "none"
    if not done:
        return None
    tests = {v: pd.read_csv(ctables / f"controls_tests{suffix}.csv") for v, suffix in VIEW_FILES.items()}
    present = [h for h in ("C1", "C2", "C3") if (tests["reg"]["hypothesis"] == h).any()]
    V["c3-status"] = "done" if "C3" in present else "pending"
    for view, t in tests.items():
        t = t[t["model"].isin(done)]
        for h in present:
            sup, cells = supported_cells(t, h)
            V[f"csup-{view}-{h.lower()}"], V[f"ccells-{view}-{h.lower()}"] = str(sup), str(cells)
            V[f"cpos-{view}-{h.lower()}"] = str(int((t[t["hypothesis"] == h]["mean_diff"] > 0).sum()))
            V[f"cnot-{view}-{h.lower()}"] = ", ".join(
                f"{MODELS[r.model][1]} on {DATASETS[r.dataset]}"
                for r in t[(t["hypothesis"] == h) & ~t["supported"]].itertuples(index=False)) or "none"
        for r in t.itertuples(index=False):
            key = f"c-{view}-{r.hypothesis.lower()}-{MODELS[r.model][0]}-{r.dataset}"
            V[key + "-ci"] = fmt_ci(r.mean_diff, r.ci_lo, r.ci_hi)
            V[key + "-p"] = fmt_p(r.p_holm)
    mslug = {"n_words": "words", "recall_minority": "recmin", "majority_amplification": "ma", "delta_p": "dp",
             "unsupported_rate": "unsup", "states_counts_share": "states", "count_wrong_share": "countwrong"}
    # The family the controls were first computed with (registered: within model x dataset), for the
    # appendix's account of the correction.
    reg_c = tests["reg"][tests["reg"]["model"].isin(done)].copy()
    reg_c["p_cell"] = reg_c.groupby(["model", "dataset"])["p"].transform(lambda x: holm(x.to_numpy()))
    V["csup-reg-c2-percell"] = str(int(((reg_c["hypothesis"] == "C2") & (reg_c["p_cell"] < 0.05)
                                        & (reg_c["mean_diff"] > 0)).sum()))
    desc = pd.read_csv(ctables / "controls_descriptive.csv")
    desc = desc[(desc["dataset"] == "both") & desc["model"].isin(done)]
    for r in desc.itertuples(index=False):
        slug = MODELS[r.model][0]
        if r.family == "count_statements":
            V[f"cd-{mslug[r.metric]}-{slug}-{cslug(r.comparison)}"] = fmt(r.mean_a, 1, percent=True)
            continue
        a, b = r.comparison.split(" vs ")
        key = f"cd-{mslug[r.metric]}-{slug}-{cslug(a)}-{cslug(b)}"
        digits = 1 if r.metric == "n_words" else 3
        V[key] = fmt_ci(r.mean_diff, r.ci_lo, r.ci_hi, digits=digits)
        V[key + "-diff"] = fmt(r.mean_diff, digits, signed=True)
    gaps = desc[(desc["metric"] == "n_words") & (desc["comparison"] == "direct_len vs kw_strat")]
    matched = gaps[gaps["mean_diff"].abs() <= 5]
    V["cd-lenmatched-models"] = ", ".join(MODELS[m][1] for m in matched["model"]) or "none"
    V["cd-lenmatched-n"] = str(len(matched))
    V["cd-lengap-min"] = fmt(gaps["mean_diff"].abs().min(), 0)
    V["cd-lengap-max"] = fmt(gaps["mean_diff"].abs().max(), 0)
    rows = []
    reg = tests["reg"]
    for m in done:
        slug, name = MODELS[m]
        for h in present:
            row = {"Model": name if h == "C1" else "", "Test": h}
            for ds, label in DATASETS.items():
                r = reg[(reg["model"] == m) & (reg["hypothesis"] == h) & (reg["dataset"] == ds)]
                row[label] = "--" if r.empty else (
                    fmt_ci(r["mean_diff"].iloc[0], r["ci_lo"].iloc[0], r["ci_hi"].iloc[0])
                    + ("$^{*}$" if bool(r["supported"].iloc[0]) else ""))
            row["Gap"] = V[f"cd-words-{slug}-directlen-kwstrat-diff"] if h == "C1" else ""
            rows.append(row)
    note = ("C1: minority coverage of \\textsf{Keywords-Grouped} minus \\textsf{Plain-Length} (above 0: grouped "
            "keywords cover more). C2: majority gap of \\textsf{Counts-Only} minus \\textsf{REFER-Counts} (above 0: "
            "the instruction to write a balanced summary reflecting this distribution pulls the summary toward an "
            "even split)." + (" C3: majority gap of \\textsf{Counts-Reflect} minus \\textsf{REFER-Counts} (above 0: "
                              "the word ``balanced'' alone does so)." if "C3" in present else "") +
            " One-sided sets; average paired difference with a 95\\% interval; $^{*}$: the difference is "
            "statistically reliable. Gap: how many words longer (+) or shorter ($-$) \\textsf{Plain-Length} "
            "summaries were than \\textsf{Keywords-Grouped} ones, the check that the length match worked.")
    return latex_table(pd.DataFrame(rows), caption="The control prompts: grouped keywords against a plain "
                       "prompt asked for the same length (C1), and the true counts given with and without the "
                       "balancing instruction (C2) or the word ``balanced'' (C3).", label="tab:controls",
                       align="llrrr", note=note,
                       size=r"\scriptsize", colsep="4pt")


def control_means(ccfg, version: str, models: list[str], V: dict) -> pd.DataFrame:
    """Condition means for the two control prompts, under the same macro names as the nine."""
    path = ccfg.scores_path
    if not path.exists():
        return pd.DataFrame()
    c = load_scores(ccfg, version)
    c = c[c["model"].isin(models)]
    scored = c[c["status"].isin(SCORED_STATUSES)]
    skew = scored[scored["level"].isin(SKEWED)]
    for cond in CONTROL_CONDITIONS:
        cc = c[c["condition"] == cond]
        V[f"ctrltrunc-{cslug(cond)}"] = fmt((cc["status"] == "truncated").mean(), 1, percent=True)
        V[f"ctrlref-{cslug(cond)}"] = str(int((cc["status"] == "refusal").sum()))
        V[f"ctrlscored-{cslug(cond)}"] = fmt(cc["status"].isin(SCORED_STATUSES).mean(), 1, percent=True)
    V["n-control-summaries"] = f"{len(c):,}"
    for m in models:
        slug = MODELS[m][0]
        for cond in CONTROL_CONDITIONS:
            sc, sk = scored[(scored["model"] == m) & (scored["condition"] == cond)], skew[
                (skew["model"] == m) & (skew["condition"] == cond)]
            for col, (mslug, skew_only) in METRICS.items():
                src = sk if skew_only else sc
                if col in src.columns and len(src):
                    V[f"mean-{mslug}-{slug}-{cslug(cond)}"] = fmt(src[col].mean(), 1 if col == "n_words" else 3)
    return c


# --- Paper figures --------------------------------------------------------------------------
# Drawn at the paper's text width (390 pt = 5.4 in) and included at 100%, so the type prints at
# the size set here. Times New Roman throughout, no titles inside the plots (the LaTeX captions
# say what each shows). Models keep one colour and one marker everywhere; colour is never the only
# cue (Okabe-Ito colours, checked for colour-vision deficiency, plus distinct markers).

FIG_W = 5.4
INK, INK2, GRID_C, BAND = "#1a1a1a", "#4a4a4a", "#e3e3e3", "#e9e9e9"
MODEL_LOOK = {"llama3.2-1b": ("#0072B2", "o"), "llama3.2-3b": ("#E69F00", "s"), "llama3.1-8b": ("#009E73", "D"),
              "qwen2.5-7b": ("#D55E00", "^"), "mistral-7b": ("#CC79A7", "v")}
MODEL_SHORT = {"llama3.2-1b": "Llama 1B", "llama3.2-3b": "Llama 3B", "llama3.1-8b": "Llama 8B",
               "qwen2.5-7b": "Qwen 7B", "mistral-7b": "Mistral 7B"}


def paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5, "legend.fontsize": 10.5, "axes.titlesize": 11,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "white",
        "figure.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.solid_capstyle": "round",
    })


def _save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)


def _model_legend(fig, models, extra=()):
    handles = [plt.Line2D([], [], linestyle="", marker=MODEL_LOOK[m][1], markersize=7, color=MODEL_LOOK[m][0],
                          markeredgecolor=INK, markeredgewidth=0.6, label=MODEL_SHORT[m]) for m in models]
    n = len(handles) + len(extra)
    fig.legend(handles=[*handles, *extra], loc="outside upper center", ncol=n if n <= 5 else 3,
               handletextpad=0.3, columnspacing=1.1)


def _dotrows(ax, rows, models, value, offsets=0.13):
    """One row per prompt, one marker per model (offset vertically), with its 95% interval."""
    ys = np.arange(len(rows))[::-1]
    for y, c in zip(ys, rows):
        for k, m in enumerate(models):
            v = value(m, c)
            if v is None:
                continue
            est, lo, hi = v
            yy = y + (k - (len(models) - 1) / 2) * offsets
            col, mk = MODEL_LOOK[m]
            if lo is not None:
                ax.plot([lo, hi], [yy, yy], color=col, linewidth=1.3, zorder=2)
            ax.plot(est, yy, marker=mk, markersize=5.2, color=col, markeredgecolor=INK, markeredgewidth=0.5,
                    linestyle="", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([pname(c) for c in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID_C, linewidth=0.7)
    ax.set_axisbelow(True)
    return ys


COVERAGE_ROWS = ["direct", "direct_len", "kw_global", "kw_strat", "kw_strat_neg"]


def fig_coverage(s: pd.DataFrame, controls: pd.DataFrame, models: list[str]) -> None:
    """RQ1: minority coverage under the plain prompt, the length-matched plain prompt and the three
    keyword prompts, for every model (one-sided sets; entity-resampled 95% intervals)."""
    ent = entity_map()
    both = pd.concat([s, controls], ignore_index=True) if controls is not None and len(controls) else s
    d = both[both["status"].isin(SCORED_STATUSES) & both["level"].isin(SKEWED)].dropna(subset=["recall_minority"])
    cache = {}
    for (m, c), g in d[d["condition"].isin(COVERAGE_ROWS)].groupby(["model", "condition"]):
        lo, hi = cluster_bootstrap_ci(g["recall_minority"].to_numpy(float), g["collection_id"].map(ent).to_numpy(),
                                      n_boot=2000)
        cache[(m, c)] = (float(g["recall_minority"].mean()), lo, hi)
    rows = [c for c in COVERAGE_ROWS if any((m, c) in cache for m in models)]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.3), layout="constrained")
    _dotrows(ax, rows, models, lambda m, c: cache.get((m, c)), offsets=0.14)
    ax.set_xlabel("Minority coverage (higher is better)")
    ax.set_xlim(0, None)
    _model_legend(fig, models)
    _save(fig, "fig_coverage.pdf")


def fig_majority_gap(d: pd.DataFrame, models: list[str], view: str, out: str) -> None:
    """RQ2: majority gap for every prompt, positive-majority sets (left) and negative-majority sets
    (right), with the band that classifier error alone could produce."""
    v = d[(d["view"] == view) & (d["dataset"] == "both") & d["model"].isin(models)]
    conds = CONDITIONS + [c for c in CONTROL_CONDITIONS if c in set(v["condition"])]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.9), sharey=True, layout="constrained")
    lo_all = min(v["ci_lo"].min(), v["envelope"].min()) - 0.01
    for ax, direction, label in zip(axes, ("positive", "negative"), ("positive majority", "negative majority")):
        sub = v[v["direction"] == direction]
        env = sub["envelope"].mean()
        ax.axvspan(env, 0, color=BAND, linewidth=0, zorder=0)
        ax.axvline(0, color=INK2, linewidth=0.9, zorder=1)
        look = {(r.model, r.condition): (r.mean, r.ci_lo, r.ci_hi) for r in sub.itertuples(index=False)}
        ys = _dotrows(ax, conds, models, lambda m, c: look.get((m, c)), offsets=0.13)
        if len(conds) > len(CONDITIONS):
            ax.axhline(ys[len(CONDITIONS) - 1] - 0.5, color=INK2, linewidth=0.6, linestyle=(0, (4, 3)))
        ax.set_xlim(lo_all, 0.03)
        ax.set_xlabel(label.capitalize(), fontsize=11, color=INK2)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.supxlabel("Majority gap (0 is proportional; grey band: within classifier error)", fontsize=12)
    _model_legend(fig, models)
    _save(fig, out)


def fig_tradeoff(V: dict, models: list[str]) -> None:
    """RQ2: what grouped keywords buy and cost. One arrow per model from the plain prompt to
    Keywords-Grouped: minority coverage (right is better) against majority gap (0 is ideal)."""
    num = lambda k: float(V[k].replace("$-$", "-"))  # noqa: E731
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6), layout="constrained")
    env = num("env-reg")
    ax.axhspan(env, 0, color=BAND, linewidth=0, zorder=0)
    ax.axhline(0, color=INK2, linewidth=0.9, zorder=1)
    ax.text(0.995, 0.004, "proportional", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=10, color=INK2, style="italic")
    for m in models:
        slug = MODELS[m][0]
        col, mk = MODEL_LOOK[m]
        x0, y0 = num(f"mean-recmin-{slug}-direct"), num(f"mean-ma-{slug}-direct")
        x1, y1 = num(f"mean-recmin-{slug}-kwstrat"), num(f"mean-ma-{slug}-kwstrat")
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, shrinkA=5, shrinkB=5, mutation_scale=12))
        ax.plot(x0, y0, marker=mk, markersize=7, color="white", markeredgecolor=col, markeredgewidth=1.6,
                linestyle="", zorder=3)
        ax.plot(x1, y1, marker=mk, markersize=7, color=col, markeredgecolor=INK, markeredgewidth=0.6,
                linestyle="", zorder=3)
    ax.set_xlabel("Minority coverage (higher is better)")
    ax.set_ylabel("Majority gap (0 is proportional)")
    ax.grid(color=GRID_C, linewidth=0.7)
    ax.set_axisbelow(True)
    extra = (plt.Line2D([], [], linestyle="", marker="o", markersize=7, color="white", markeredgecolor=INK2,
                        markeredgewidth=1.4, label="Plain"),
             plt.Line2D([], [], linestyle="", marker="o", markersize=7, color=INK2, label="Keywords-Grouped"))
    _model_legend(fig, models)
    ax.legend(handles=list(extra), loc="lower left", handletextpad=0.3)
    _save(fig, "fig_tradeoff.pdf")


def fig_controls(ccfg, models: list[str]) -> None:
    """The two control tests per model and review source: C1 (left) and C2 (right), mean paired
    difference with entity-resampled 95% interval; above zero supports the test."""
    path = ccfg.results_dir / "tables" / "controls_tests.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    t = t[t["model"].isin(models)]
    looks = {"amazon": ("#0072B2", "o", "Amazon"), "drugs": ("#D55E00", "^", "Drugs.com")}
    hyps = [h for h in ("C1", "C2", "C3") if (t["hypothesis"] == h).any()]
    fig, axes = plt.subplots(1, len(hyps), figsize=(FIG_W, 3.0), sharey=True, layout="constrained", squeeze=False)
    axes = axes[0]
    ys = np.arange(len(models))[::-1]
    labels = {"C1": "Coverage gain\nover Plain-Length", "C2": "Dropping the whole\nbalancing instruction",
              "C3": "Dropping only\n“balanced”"}
    for ax, h in zip(axes, hyps):
        ax.axvline(0, color=INK2, linewidth=0.9)
        for y, m in zip(ys, models):
            for k, (ds, (col, mk, _)) in enumerate(looks.items()):
                r = t[(t["model"] == m) & (t["hypothesis"] == h) & (t["dataset"] == ds)]
                if r.empty:
                    continue
                r = r.iloc[0]
                yy = y + (0.13 if k == 0 else -0.13)
                ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=col, linewidth=1.4)
                ax.plot(r["mean_diff"], yy, marker=mk, markersize=6.5, color=col if r["supported"] else "white",
                        markeredgecolor=col if not r["supported"] else INK, markeredgewidth=1.0, linestyle="")
        ax.set_yticks(ys)
        ax.set_yticklabels([MODEL_SHORT[m] for m in models])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.grid(axis="x", color=GRID_C, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel(labels[h])
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    handles = [plt.Line2D([], [], linestyle="", marker=mk, markersize=7, color=col, markeredgecolor=INK,
                          markeredgewidth=0.6, label=lab) for col, mk, lab in looks.values()]
    handles.append(plt.Line2D([], [], linestyle="", marker="o", markersize=7, color="white",
                              markeredgecolor=INK2, markeredgewidth=1.0, label="not reliable"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, handletextpad=0.3, columnspacing=1.2)
    _save(fig, "fig_controls.pdf")


def fig_faith(V: dict, models: list[str]) -> None:
    """RQ4: the attribution problem. Unsupported rate per prompt (average over models) with the
    planned check and with the corrected check that also tests statements without ``some reviewers
    say''; the thin markers are the individual models."""
    num = lambda k: float(V[k].replace("$-$", "-")) if k in V and V[k] != "--" else None  # noqa: E731
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6), layout="constrained")
    ys = np.arange(len(CONDITIONS))[::-1]
    for y, c in zip(ys, CONDITIONS):
        reg = [num(f"mean-unsup-{MODELS[m][0]}-{cslug(c)}") for m in models]
        cor = [num(f"mean-unsupd-{MODELS[m][0]}-{cslug(c)}") for m in models]
        reg, cor = [v for v in reg if v is not None], [v for v in cor if v is not None]
        if not reg or not cor:
            continue
        a, b = float(np.mean(reg)), float(np.mean(cor))
        ax.plot([b, a], [y, y], color=INK2, linewidth=2.0, zorder=1)
        ax.plot(reg, [y + 0.22] * len(reg), linestyle="", marker="|", markersize=7, color="#9a9a9a", zorder=1)
        ax.plot(cor, [y - 0.22] * len(cor), linestyle="", marker="|", markersize=7, color="#0072B2", alpha=0.6,
                zorder=1)
        ax.plot(a, y, marker="o", markersize=8, color="white", markeredgecolor=INK, markeredgewidth=1.4,
                linestyle="", zorder=3)
        ax.plot(b, y, marker="o", markersize=8, color="#0072B2", markeredgecolor=INK, markeredgewidth=0.6,
                linestyle="", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([pname(c) for c in CONDITIONS])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(CONDITIONS) - 0.4)
    ax.grid(axis="x", color=GRID_C, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of summary statements that no review supports")
    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=8, color="white", markeredgecolor=INK,
                          markeredgewidth=1.4, label="planned check"),
               plt.Line2D([], [], linestyle="", marker="o", markersize=8, color="#0072B2", markeredgecolor=INK,
                          markeredgewidth=0.6, label="corrected check"),
               plt.Line2D([], [], linestyle="", marker="|", markersize=9, color="#9a9a9a", label="single models")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, handletextpad=0.3)
    _save(fig, "fig_faith.pdf")


FAMILY_LOOK = {"direct": ("#6b6b6b", "o"), "kw_global": ("#E69F00", "s"), "kw_strat": ("#D55E00", "D"),
               "refer": ("#0072B2", "^"), "refer_oracle": ("#56B4E9", "v"), "counts_neutral": ("#CC79A7", "P"), "counts_reflect": ("#9F5A8A", "*"),
               "direct_len": ("#999999", "X"), "kw_strat_neg": ("#F0A35E", "d"), "cot_refer": ("#2B5C8A", "<"),
               "key_refer": ("#009E73", ">"), "key_oracle": ("#7FC9B0", "h")}


def fig_length_hist(s: pd.DataFrame, controls: pd.DataFrame, models: list[str]) -> None:
    """Histogram of summary length: keyword prompts write longer summaries, which is why coverage is
    also compared at equal length and against the length-matched plain prompt."""
    both = pd.concat([s, controls], ignore_index=True) if controls is not None and len(controls) else s
    d = both[both["status"].isin(SCORED_STATUSES) & both["model"].isin(models)]
    show = [c for c in ("direct", "kw_global", "kw_strat", "direct_len") if c in set(d["condition"])]
    bins = np.arange(0, 241, 10)
    fig, ax = plt.subplots(figsize=(FIG_W, 3.1), layout="constrained")
    for c in show:
        col = FAMILY_LOOK[c][0]
        w = d.loc[d["condition"] == c, "n_words"].clip(upper=239)
        ax.hist(w, bins=bins, histtype="stepfilled", color=col, alpha=0.18, linewidth=0)
        ax.hist(w, bins=bins, histtype="step", color=col, linewidth=1.8, label=pname(c),
                linestyle=(0, (4, 2)) if c == "direct_len" else "-")
    ax.set_xlabel("Summary length (words)")
    ax.set_ylabel("Number of summaries")
    ax.grid(axis="y", color=GRID_C, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.legend(loc="outside upper center", ncol=len(show), handlelength=1.6, columnspacing=1.2)
    _save(fig, "fig_length.pdf")


LINE_PROMPTS = ["direct", "kw_global", "kw_strat", "refer", "refer_oracle", "counts_neutral"]
TRUE_NEG = {"pos88": 0.125, "pos75": 0.25, "bal": 0.5, "neg75": 0.75, "neg88": 0.875}


def fig_proportion_line(s: pd.DataFrame, controls: pd.DataFrame, models: list[str]) -> None:
    """Line graph: the share of negative opinion in the summary against the true share in the reviews,
    averaged over models. A proportional summary lies on the diagonal; a flatter line means summaries
    drift toward an even split."""
    both = pd.concat([s, controls], ignore_index=True) if controls is not None and len(controls) else s
    d = both[both["status"].isin(SCORED_STATUSES) & both["model"].isin(models)].dropna(subset=["summary_neg_share"])
    fig, ax = plt.subplots(figsize=(FIG_W, 3.9), layout="constrained")
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1.0, linestyle=(0, (5, 4)), zorder=1)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":", zorder=1)
    ax.text(0.97, 0.985, "proportional", ha="right", va="top", fontsize=10, color=INK2, style="italic")
    ax.text(0.02, 0.51, "even split", ha="left", va="bottom", fontsize=10, color=INK2, style="italic")
    xs = list(TRUE_NEG.values())
    for c in [p for p in LINE_PROMPTS if p in set(d["condition"])]:
        g = d[d["condition"] == c]
        # average over models first, so every model counts equally
        per_model = g.groupby(["model", "level"])["summary_neg_share"].mean().unstack()
        ys = [per_model[lv].mean() for lv in TRUE_NEG]
        col, mk = FAMILY_LOOK[c]
        ax.plot(xs, ys, color=col, marker=mk, markersize=6.5, linewidth=1.8, markeredgecolor=INK,
                markeredgewidth=0.5, label=pname(c), zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["1/8", "2/8", "4/8", "6/8", "7/8"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Share of negative reviews in the set")
    ax.set_ylabel("Negative share of the summary")
    ax.grid(color=GRID_C, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.legend(loc="outside upper center", ncol=3, handlelength=1.8, columnspacing=1.2)
    _save(fig, "fig_proportion.pdf")


def fig_heatmap_gap(d: pd.DataFrame, models: list[str], env: float) -> None:
    """Heat map: majority gap for every prompt (rows) and model (columns), positive-majority sets left,
    negative-majority sets right. Blue is close to proportional, red means the summary looks more
    balanced than the reviews; the colour bar marks the limit classifier error alone could explain."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    v = d[(d["view"] == "registered") & (d["dataset"] == "both") & d["model"].isin(models)]
    conds = CONDITIONS + [c for c in CONTROL_CONDITIONS if c in set(v["condition"])]
    cmap = LinearSegmentedColormap.from_list("gap", ["#8c1d18", "#d6604d", "#f7f7f7", "#4393c3", "#1f4e79"])
    lo = min(-0.25, float(v["mean"].min()))
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=-lo)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.9), sharey=True, layout="constrained")
    for ax, direction in zip(axes, ("positive", "negative")):
        sub = v[v["direction"] == direction]
        grid = np.array([[float(sub[(sub["condition"] == c) & (sub["model"] == m)]["mean"].iloc[0])
                          for m in models] for c in conds])
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                txt = f"{val:+.2f}".replace("+0.", "+.").replace("-0.", "\u2212.")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        color="white" if abs(norm(val) - 0.5) > 0.33 else INK)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_SHORT[m].replace(" ", "\n") for m in models], fontsize=9.5)
        ax.set_xlabel(f"{direction.capitalize()} majority", fontsize=11.5)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        if len(conds) > len(CONDITIONS):
            ax.axhline(len(CONDITIONS) - 0.5, color="white", linewidth=3)
    axes[0].set_yticks(range(len(conds)))
    axes[0].set_yticklabels([pname(c) for c in conds])
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.8, aspect=35, pad=0.02)
    cb.set_label("Majority gap (0 = proportional)")
    cb.ax.axvline(env, color=INK, linewidth=1.2)
    cb.ax.tick_params(labelsize=9.5)
    cb.outline.set_visible(False)
    _save(fig, "fig_heatmap.pdf")


def stated_split_categories(cfg, models: list[str], V: dict) -> dict[str, int]:
    """What split REFER summaries state on one-sided sets, pooled over models."""
    counts = Counter()
    for g in read_jsonl(cfg.generations_path):
        if g["model"] not in models or g["condition"] != "refer" or g["level"] not in SKEWED:
            continue
        stated = split_summary(g["raw"], drop_unterminated_tail=g["status"] == "truncated").counts
        pos = next((n for n, p in stated if p == "positive" and 0 <= n <= 8), None)
        neg = next((n for n, p in stated if p == "negative" and 0 <= n <= 8), None)
        if pos is None or neg is None or pos + neg == 0:
            continue
        majority = pos if g["level"].startswith("pos") else neg
        share, true = majority / (pos + neg), TRUE_MAJORITY_SHARE[g["level"]]
        key = ("correct" if abs(share - true) < 1e-9 else "even" if pos == neg
               else "under" if share < true else "over")
        counts[key] += 1
    n = sum(counts.values())
    V["split-n"] = f"{n:,}"
    for k in ("correct", "even", "under", "over"):
        V[f"split-{k}"] = fmt(counts[k] / n if n else float("nan"), 0, percent=True)
    return dict(counts)


def fig_stated_splits(counts: dict[str, int]) -> None:
    """Donut: the splits REFER summaries state on one-sided sets."""
    labels = {"correct": "True split", "even": "Even 4:4", "under": "Majority understated\n(not 4:4)",
              "over": "Majority overstated"}
    colours = {"correct": "#009E73", "even": "#E69F00", "under": "#D55E00", "over": "#56B4E9"}
    keys = [k for k in ("correct", "even", "under", "over") if counts.get(k)]
    vals = [counts[k] for k in keys]
    n = sum(vals)
    fig, ax = plt.subplots(figsize=(FIG_W, 3.3), layout="constrained")
    wedges, _ = ax.pie(vals, colors=[colours[k] for k in keys], startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    ax.text(0, 0.06, f"{n:,}", ha="center", va="center", fontsize=17, color=INK)
    ax.text(0, -0.16, "stated splits", ha="center", va="center", fontsize=10.5, color=INK2)
    for w, k, v in zip(wedges, keys, vals):
        ang = np.deg2rad((w.theta2 + w.theta1) / 2)
        x, y = np.cos(ang), np.sin(ang)
        ax.annotate(f"{labels[k]}\n{100 * v / n:.0f}%", xy=(0.82 * x, 0.82 * y), xytext=(1.32 * x, 1.18 * y),
                    ha="left" if x >= 0 else "right", va="center", fontsize=10.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_aspect("equal")
    _save(fig, "fig_splits.pdf")


RADAR_PROMPTS = ["direct", "kw_global", "kw_strat", "refer"]


def fig_radar(V: dict, models: list[str]) -> None:
    """Radar chart: the profile of the main prompts, averaged over models. Each axis runs from the
    lowest to the highest value among all nine prompts, with the better end outside."""
    num = lambda k: float(V[k].replace("$-$", "-"))  # noqa: E731
    mean = lambda key, c: float(np.mean([num(f"mean-{key}-{MODELS[m][0]}-{cslug(c)}") for m in models]))  # noqa: E731
    axes_def = [("Minority\ncoverage", lambda c: mean("recmin", c)),
                ("Proportion\nkept", lambda c: 1 - abs(mean("ma", c))),
                ("Supported\nstatements", lambda c: 1 - mean("unsupd", c)),
                ("Not\ncontradicted", lambda c: 1 - mean("contrad", c))]
    raw = {c: [f(c) for _, f in axes_def] for c in CONDITIONS}
    lo = [min(raw[c][i] for c in CONDITIONS) for i in range(len(axes_def))]
    hi = [max(raw[c][i] for c in CONDITIONS) for i in range(len(axes_def))]
    scale = lambda i, v: 0.15 + 0.85 * (v - lo[i]) / ((hi[i] - lo[i]) or 1)  # noqa: E731
    angles = np.linspace(0, 2 * np.pi, len(axes_def), endpoint=False).tolist()
    fig = plt.figure(figsize=(FIG_W, 4.3), layout="constrained")
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for c in RADAR_PROMPTS:
        r = [scale(i, v) for i, v in enumerate(raw[c])]
        col, mk = FAMILY_LOOK[c]
        ax.plot(angles + angles[:1], r + r[:1], color=col, linewidth=2.0, marker=mk, markersize=6,
                markeredgecolor=INK, markeredgewidth=0.5, label=pname(c))
        ax.fill(angles + angles[:1], r + r[:1], color=col, alpha=0.08)
    ax.set_xticks(angles)
    ax.set_xticklabels([a for a, _ in axes_def], fontsize=11)
    ax.tick_params(axis="x", pad=16)
    ax.set_yticks([0.15, 0.575, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID_C, linewidth=0.8)
    ax.spines["polar"].set_visible(False)
    fig.legend(loc="outside upper center", ncol=4, handlelength=1.8, columnspacing=1.2)
    _save(fig, "fig_radar.pdf")


def wordcloud_counts(cfg, models: list[str], V: dict) -> Counter:
    """Minority keywords (from the negative reviews of positive-majority drug sets) that appear in a
    model's Keywords-Grouped summary but not in its plain summary of the same set."""
    kws = {r["collection_id"]: r for r in read_jsonl(KEYWORDS_PATH)}
    majority_text = {c.collection_id: " ".join(t.lower() for t, pol in zip(c.texts, c.polarities) if pol == "positive")
                     for c in read_collections(COLLECTIONS_PATH) if c.dataset == "drugs"}
    generic = {"pill", "pills", "medication", "medicine", "dose", "doses", "dosage", "week", "weeks", "month",
               "months", "day", "days", "year", "years", "night", "nights", "time", "times", "doctor", "drug"}
    text = {}
    for g in read_jsonl(cfg.generations_path):
        if (g["model"] in models and g["condition"] in ("direct", "kw_strat") and g["dataset"] == "drugs"
                and g["level"] in ("pos75", "pos88")):
            text[(g["model"], g["collection_id"], g["condition"])] = g["raw"].lower()
    counts = Counter()
    for (m, cid, cond), t in text.items():
        if cond != "kw_strat" or (m, cid, "direct") not in text:
            continue
        plain = text[(m, cid, "direct")]
        for phrase in kws[cid]["strat"].get("negative", []):
            p = phrase.lower()
            if p in generic or p in majority_text.get(cid, ""):
                continue  # keep only what the minority reviews say
            if p in t and p not in plain:
                counts[p] += 1
    V["wc-phrases"] = f"{len(counts):,}"
    V["wc-mentions"] = f"{sum(counts.values()):,}"
    return counts


def fig_wordcloud(counts: Counter) -> None:
    """Word cloud of the minority phrases grouped keywords brought into drug summaries."""
    from wordcloud import WordCloud
    if not counts:
        return
    shades = ["#8c1d18", "#b2361f", "#d55e00", "#a8441c", "#6e1a12"]
    colour = lambda word, **_: shades[hash(word) % len(shades)]  # noqa: E731
    wc = WordCloud(width=2000, height=900, background_color="white", font_path="C:/Windows/Fonts/times.ttf",
                   prefer_horizontal=0.95, max_words=70, relative_scaling=0.5, margin=6, random_state=7,
                   color_func=colour, collocations=False).generate_from_frequencies(dict(counts))
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 900 / 2000), layout="constrained")
    ax.imshow(wc.to_array(), interpolation="bilinear")
    ax.axis("off")
    fig.savefig(FIG / "fig_wordcloud.pdf", dpi=300)
    plt.close(fig)


def human_check_numbers(cfg, V: dict) -> None:
    path = cfg.results_dir / "human_check.json"
    V["human-check-done"] = "yes" if path.exists() else "no"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    choice = report.get("primary_faithfulness", {})
    if choice:
        V["human-primary"] = choice["choice"]
        V["human-alpha-reg"] = fmt(choice["alpha_as_written"])
        V["human-alpha-deattr"] = fmt(choice["alpha_deattr"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scorer-version", default=SCORER_VERSION)
    ap.add_argument("--controls", default="configs/kw_controls.toml",
                    help="the amendment-9 control run (its tables come from kw_amendment_analysis.py)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    s = load_scores(cfg, args.scorer_version)
    expected = 60 * 5 * 2 * len(cfg.conditions)
    counts = s.groupby("model").size()
    complete = [m for m in MODELS if counts.get(m, 0) == expected]
    pending = [m for m in MODELS if m not in complete]
    print(f"scorer v{args.scorer_version}: complete {complete}; pending {pending} "
          f"({ {m: int(counts.get(m, 0)) for m in pending} })")

    V: dict[str, str] = {"scorer-version": str(args.scorer_version),
                         "n-models-complete": str(len(complete)),
                         "models-complete": ", ".join(MODELS[m][1] for m in complete) or "none",
                         "models-pending": ", ".join(MODELS[m][1] for m in pending) or "none"}
    data_numbers(cfg, V)
    instrument_numbers(cfg, V)
    human_check_numbers(cfg, V)
    s = s[s["model"].isin(complete)]
    if complete:
        manip = s.drop_duplicates("collection_id").groupby("dataset")["negation_changed"].mean()
        for ds, v in manip.items():
            V[f"negchanged-{ds}"] = fmt(v, 1, percent=True)
        for m in complete:
            model_numbers(s, m, V)
            stated_split_numbers(cfg, m, V)
            (GEN / f"tab_means_{MODELS[m][0]}.tex").write_text(means_table(m, V), encoding="utf-8")
        views = {"reg": run_tests(s), "pol": run_tests(polarity_only_view(s))}
        # Only when the chosen scorer version computed it: older rows carry the column empty.
        if "unsupported_rate_deattr" in s.columns and s["unsupported_rate_deattr"].notna().any():
            views["deattr"] = run_tests(deattributed_view(s))
        hypothesis_numbers(views, V)
        pooled_numbers(s, views, V)
        (GEN / "tab_hyp_matrix.tex").write_text(hypothesis_matrix(views, complete), encoding="utf-8")
        for m in complete:
            (GEN / f"tab_hyp_{MODELS[m][0]}.tex").write_text(hypothesis_full(views, m), encoding="utf-8")
        (GEN / "tab_counts.tex").write_text(counts_table(V, complete), encoding="utf-8")
        for which in CROSS_TABLES:
            (GEN / f"tab_cross_{which}.tex").write_text(cross_model_table(V, complete, which), encoding="utf-8")
        for view in ROBUST_VIEWS:
            if view in views:
                (GEN / f"tab_robust_{view}.tex").write_text(robustness_table(views, view, complete),
                                                            encoding="utf-8")
        # Amendment 9: controls, entity-clustered tests, majority direction.
        ccfg = load_config(args.controls)
        controls = control_means(ccfg, args.scorer_version, complete, V)
        ctable = control_numbers(ccfg, complete, V)
        if ctable:
            (GEN / "tab_controls.tex").write_text(ctable, encoding="utf-8")
        (GEN / "tab_cluster.tex").write_text(clustered_numbers(cfg, views, complete, V), encoding="utf-8")
        directions = direction_numbers(cfg, complete, V)
        abs_distortion_numbers(cfg, complete, V)
        (GEN / "tab_length.tex").write_text(length_numbers(s, complete, V, controls), encoding="utf-8")
        (GEN / "appendix_tables.tex").write_text(appendix_tables(complete, views), encoding="utf-8")
        (GEN / "tab_compliance.tex").write_text(compliance_table(V, complete), encoding="utf-8")
        paper_style()
        fig_coverage(s, controls, complete)
        fig_majority_gap(directions, complete, "registered", "fig_ma.pdf")
        fig_majority_gap(directions, complete, "polarity_only", "fig_ma_alt.pdf")
        fig_tradeoff(V, complete)
        fig_controls(ccfg, complete)
        fig_faith(V, complete)
        fig_length_hist(s, controls, complete)
        fig_proportion_line(s, controls, complete)
        fig_heatmap_gap(directions, complete, float(V["env-reg"].replace("$-$", "-")))
        fig_stated_splits(stated_split_categories(cfg, complete, V))
        fig_radar(V, complete)
        fig_wordcloud(wordcloud_counts(cfg, complete, V))
    # The labelling rules, verbatim as the annotation page prints them (from "claim" on; the
    # steps before that are the spreadsheet route's, and the page gives its own).
    wrapped = []
    for line in GUIDE[GUIDE.index("claim"):]:
        indent = " " * (len(line) - len(line.lstrip()) + 4)
        wrapped += textwrap.wrap(line, 92, subsequent_indent=indent) or [""]
    (GEN / "guide.tex").write_text("{\\scriptsize\n\\begin{verbatim}\n" + "\n".join(wrapped)
                                   + "\n\\end{verbatim}\n}\n", encoding="utf-8")
    (GEN / "numbers.tex").write_text(macro_file(V), encoding="utf-8")
    print(f"{len(V)} numbers, tables in {GEN}, figures in {FIG}")


if __name__ == "__main__":
    main()
