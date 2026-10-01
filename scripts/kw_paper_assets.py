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
from fairsum.kwsum.paper import fmt, fmt_ci, fmt_p, latex_table, macro_file
from fairsum.kwsum.propositions import split_summary
from fairsum.kwsum.scoring import SCORER_VERSION
from fairsum.kwsum.stats import bootstrap_ci, deattributed_view, polarity_only_view, run_tests
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


def code(condition: str) -> str:
    return r"\texttt{" + condition.replace("_", r"\_") + "}"


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
    km = pd.DataFrame(rows).groupby(["dataset", "level"])["kw"].mean()
    for (ds, level), v in km.items():
        V[f"kwmin-{ds}-{level}"] = fmt(v)
    review_share = {"pos75": 0.25, "neg75": 0.25, "pos88": 0.125, "neg88": 0.125}
    kw_table = pd.DataFrame([{"Dataset": DATASETS[ds], "Level": LEVELS[level],
                              "Minority share of reviews": fmt(review_share[level]),
                              "Minority share of keywords": fmt(v)}
                             for (ds, level), v in km.items()])
    (GEN / "tab_kwmin.tex").write_text(latex_table(
        kw_table, caption="Does keyword selection under-sample the minority? Share of \\texttt{kw\\_global} "
        "keyphrases found in at least one minority review, test collections.", label="tab:kwmin",
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


def stated_split_numbers(cfg, m: str, V: dict) -> None:
    """The positive:negative split REFER outputs state on skewed inputs, and how often it is 4:4."""
    slug = MODELS[m][0]
    for c in ("refer", "cot_refer"):
        pairs = []
        for g in read_jsonl(cfg.generations_path):
            if g["model"] != m or g["condition"] != c or g["level"] not in SKEWED:
                continue
            counts = split_summary(g["raw"], drop_unterminated_tail=g["status"] == "truncated").counts
            pos = next((n for n, p in counts if p == "positive" and 0 <= n <= 8), None)
            neg = next((n for n, p in counts if p == "negative" and 0 <= n <= 8), None)
            if pos is not None and neg is not None:
                pairs.append((pos, neg))
        if not pairs:
            continue
        (mode, freq), = Counter(pairs).most_common(1)
        V[f"statedmode-{slug}-{cslug(c)}"] = f"{mode[0]}:{mode[1]}"
        V[f"statedmodeshare-{slug}-{cslug(c)}"] = fmt(freq / len(pairs), 0, percent=True)
        V[f"statedn-{slug}-{cslug(c)}"] = str(len(pairs))
        V[f"statedeven-{slug}-{cslug(c)}"] = fmt(sum(p == n for p, n in pairs) / len(pairs), 0, percent=True)


def means_table(m: str, V: dict) -> str:
    slug, name = MODELS[m]
    has_d = f"mean-unsupd-{slug}-direct" in V and V[f"mean-unsupd-{slug}-direct"] != "--"
    rows = []
    for c in CONDITIONS:
        g = lambda k: V.get(f"mean-{k}-{slug}-{cslug(c)}", "--")  # noqa: E731
        row = {"Prompt": code(c), "Unsup.": g("unsup")}
        if has_d:
            row["Unsup.$_{\\mathrm{robust}}$"] = g("unsupd")
        row.update({"Contra.": g("contra"), "$|\\Delta p|$": g("dp"), "MA": g("ma"),
                    "Min.\\ cov.": g("recmin"), "Words": g("words")})
        rows.append(row)
    df = pd.DataFrame(rows)
    note = ("Means over both datasets. Unsup.: share of summary statements entailed by no single review "
            "(registered metric)" + ("; Unsup.$_{\\mathrm{robust}}$: the same with reviewer attribution "
                                     "removed (Section~\\ref{sec:faithfulness})" if has_d else "") +
            ". Contra.: contradicted and unsupported. $|\\Delta p|$: absolute gap between the summary's "
            "and the source's negative share (all levels). MA: majority amplification; Min.\\ cov.: share of "
            "minority reviews that support at least one statement (skewed levels).")
    return latex_table(df, caption=f"Condition means for {name}.", label=f"tab:means-{slug}",
                       align="l" + "r" * (len(df.columns) - 1), note=note, colsep="3pt")


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
    note = ("A: supported on Amazon; D: on Drugs.com; --: on neither (Holm-corrected paired tests; "
            "non-inferiority by the 95\\% interval). $^{\\dagger}$: the verdict changes in the robustness "
            "analysis (polarity model alone for proportion metrics; attribution-robust NLI for faithfulness "
            "and coverage). n/a: registered for another model only.")
    return latex_table(pd.DataFrame(rows), caption="Registered hypothesis tests: where each is supported.",
                       label="tab:hyp-matrix", align="l" + "c" * len(models), note=note)


SHORT_METRIC = {"unsupported_rate": "unsupported", "majority_amplification": "MA",
                "coverage_gap": "coverage gap", "contradiction_rate": "contradiction",
                "count_abs_error": "count error", "delta_p": "$|\\Delta p|$"}


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
    return latex_table(pd.DataFrame(rows), caption=f"Registered tests for {MODELS[m][1]}: mean paired "
                       "difference (first minus second prompt of Table~\\ref{tab:hypotheses}) with 95\\% "
                       "bootstrap interval, and Holm-corrected $p$ (A: Amazon, D: Drugs.com).",
                       label=f"tab:hyp-{MODELS[m][0]}", align="llrrrr", size=r"\scriptsize",
                       colsep="3pt")


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
    return latex_table(pd.DataFrame(rows), caption="Stated review counts: how often prompts that ask for "
                       "counting state them, and how often the stated counts are exact, by level "
                       "(positive:negative). Number of outputs stating counts in parentheses.",
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
    return latex_table(pd.DataFrame(rows), caption="Output length: share truncated at 256 tokens / mean "
                       "words.", label="tab:compliance", align="l" + "r" * len(models), size=r"\small")


def fig_ma(s: pd.DataFrame, models: list[str]) -> None:
    scored = s[s["status"].isin(SCORED_STATUSES) & s["level"].isin(SKEWED)]
    fig, axes = plt.subplots(len(models), 2, figsize=(7.2, 1.9 + 2.2 * len(models)), squeeze=False,
                             sharex=True)
    ys = np.arange(len(CONDITIONS))[::-1]
    for i, m in enumerate(models):
        for j, ds in enumerate(DATASETS):
            ax = axes[i][j]
            ax.axvline(0, color=F.INK2, linewidth=1, linestyle=(0, (3, 3)))
            for y, c in zip(ys, CONDITIONS):
                sub = scored[(scored["model"] == m) & (scored["dataset"] == ds) & (scored["condition"] == c)]
                for col, face, off in (("majority_amplification", F.SLOTS[0], 0.15),
                                       ("majority_amplification_polarity_only", F.SURFACE, -0.15)):
                    if col not in sub.columns:
                        continue
                    v = sub[col].dropna().to_numpy()
                    lo, hi = bootstrap_ci(v, n_boot=2000)
                    ax.plot([lo, hi], [y + off, y + off], color=F.SLOTS[0], linewidth=1.6)
                    ax.plot(v.mean(), y + off, marker="o", markersize=5, color=F.SLOTS[0],
                            markerfacecolor=face, markeredgewidth=1.4, linestyle="")
            ax.set_yticks(ys)
            ax.set_yticklabels(CONDITIONS if j == 0 else [], fontsize=8)
            ax.set_title(f"{MODELS[m][1]} · {DATASETS[ds]}", fontsize=9, color=F.INK)
            ax.grid(axis="x", color=F.GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", color=F.SLOTS[0], linestyle="", label="hybrid instrument"),
               plt.Line2D([], [], marker="o", color=F.SLOTS[0], markerfacecolor=F.SURFACE, linestyle="",
                          label="polarity model alone")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.supxlabel("Majority amplification on skewed inputs (< 0: majority under-represented)", fontsize=9,
                  color=F.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "fig_ma.pdf", facecolor=F.SURFACE)
    plt.close(fig)


def fig_tradeoff(V: dict, models: list[str]) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.2), squeeze=False)
    for ax, m in zip(axes[0], models):
        slug = MODELS[m][0]
        for c in CONDITIONS:
            x = float(V[f"mean-recmin-{slug}-{cslug(c)}"])
            y = float(V[f"mean-dp-{slug}-{cslug(c)}"].replace("$-$", "-"))
            ax.scatter(x, y, s=38, color=F.FAMILY_COLOR[F.FAMILY[c]], edgecolor=F.SURFACE, linewidth=1.2,
                       zorder=3)
            # kw_strat and kw_strat_neg sit close together: label the first below its point.
            offset = (4, -9) if c == "kw_strat" else (4, 3)
            ax.annotate(c, (x, y), xytext=offset, textcoords="offset points", fontsize=7, color=F.INK2)
        ax.set_title(MODELS[m][1], fontsize=9, color=F.INK)
        ax.set_xlabel("Minority coverage (higher is better)", fontsize=8, color=F.INK2)
        ax.set_ylabel("$|\\Delta p|$ (lower is better)", fontsize=8, color=F.INK2)
        ax.grid(color=F.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=F.FAMILY_COLOR[f], label=f)
               for f in ("baseline", "REFER", "keywords", "combined")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig_tradeoff.pdf", facecolor=F.SURFACE)
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
        (GEN / "tab_hyp_matrix.tex").write_text(hypothesis_matrix(views, complete), encoding="utf-8")
        for m in complete:
            (GEN / f"tab_hyp_{MODELS[m][0]}.tex").write_text(hypothesis_full(views, m), encoding="utf-8")
        (GEN / "tab_counts.tex").write_text(counts_table(V, complete), encoding="utf-8")
        (GEN / "tab_compliance.tex").write_text(compliance_table(V, complete), encoding="utf-8")
        fig_ma(s, complete)
        fig_tradeoff(V, complete)
    # The annotators' guide, verbatim from the tool that shows it to them.
    wrapped = []
    for line in GUIDE:
        indent = " " * (len(line) - len(line.lstrip()) + 4)
        wrapped += textwrap.wrap(line, 92, subsequent_indent=indent) or [""]
    (GEN / "guide.tex").write_text("{\\scriptsize\n\\begin{verbatim}\n" + "\n".join(wrapped)
                                   + "\n\\end{verbatim}\n}\n", encoding="utf-8")
    (GEN / "numbers.tex").write_text(macro_file(V), encoding="utf-8")
    print(f"{len(V)} numbers, tables in {GEN}, figures in {FIG}")


if __name__ == "__main__":
    main()
