"""Paper figures. Colours and mark specs follow the dataviz reference palette:
fixed categorical slots, 2px lines, recessive grid, text in ink rather than
series colour, direct labels, one y-axis per chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fairsum.kwsum.compliance import SCORED_STATUSES  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order

LINE_CONDITIONS = ("direct", "kw_global", "kw_strat_neg", "refer")
FAMILY = {
    "direct": "baseline", "refer": "REFER", "cot_refer": "REFER", "refer_oracle": "REFER",
    "kw_global": "keywords", "kw_strat": "keywords", "kw_strat_neg": "keywords",
    "key_refer": "combined", "key_oracle": "combined",
}
FAMILY_COLOR = {"REFER": SLOTS[0], "keywords": SLOTS[1], "combined": SLOTS[2], "baseline": MUTED}
MINORITY_SHARE = {"pos75": "25%", "neg75": "25%", "pos88": "12.5%", "neg88": "12.5%"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _scored(scores: pd.DataFrame) -> pd.DataFrame:
    return scores[scores["status"].isin(SCORED_STATUSES)]


def fig_majority_amplification(scores: pd.DataFrame, path: str | Path) -> None:
    """Mean majority amplification at 25% vs 12.5% minority share (H1's dose-response)."""
    s = _scored(scores)
    s = s[s["majority_amplification"].notna() & s["condition"].isin(LINE_CONDITIONS)].copy()
    s["minority_share"] = s["level"].map(MINORITY_SHARE)
    panels = sorted(s.groupby(["dataset", "model"]).groups)
    if not panels:
        return
    datasets = sorted({d for d, _ in panels})
    models = sorted({m for _, m in panels})
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(3.2 * len(models), 2.8 * len(datasets)),
                             squeeze=False, sharey=True)
    for r, dataset in enumerate(datasets):
        for c, model in enumerate(models):
            ax = axes[r][c]
            _style(ax)
            ax.axhline(0, color=AXIS, linewidth=1)
            sub = s[(s["dataset"] == dataset) & (s["model"] == model)]
            for i, cond in enumerate(LINE_CONDITIONS):
                means = sub[sub["condition"] == cond].groupby("minority_share")["majority_amplification"].mean()
                ys = [means.get("25%"), means.get("12.5%")]
                if any(y is None for y in ys):
                    continue
                ax.plot([0, 1], ys, color=SLOTS[i], linewidth=2, marker="o", markersize=6, label=cond)
                ax.annotate(cond, (1, ys[1]), xytext=(5, 0), textcoords="offset points",
                            color=INK2, fontsize=7, va="center")
            ax.set_xticks([0, 1], ["25%", "12.5%"])
            ax.set_xlim(-0.15, 1.6)
            ax.set_title(f"{dataset} · {model}", color=INK, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
    fig.supxlabel("Minority share of the 8 reviews", color=INK2)
    fig.supylabel("Majority amplification (> 0: majority over-represented)", color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(scores: pd.DataFrame, path: str | Path) -> None:
    """Faithfulness (unsupported rate) against proportional fairness (|dp|), per condition."""
    s = _scored(scores)
    means = s.groupby(["model", "condition"])[["unsupported_rate", "delta_p"]].mean().reset_index()
    models = sorted(means["model"].unique())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for _, row in means[means["model"] == model].iterrows():
            color = FAMILY_COLOR[FAMILY[row["condition"]]]
            ax.scatter(row["unsupported_rate"], row["delta_p"], s=40, color=color,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(row["condition"], (row["unsupported_rate"], row["delta_p"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
        ax.set_title(model, color=INK, fontsize=9)
        ax.set_xlabel("Unsupported-proposition rate (lower is more faithful)")
    axes[0][0].set_ylabel("|Δp| sentiment-share error (lower is fairer)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=FAMILY_COLOR[f], label=f)
               for f in ("baseline", "REFER", "keywords", "combined")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
