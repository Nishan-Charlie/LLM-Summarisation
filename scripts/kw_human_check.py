"""Human check of the scorers on ~100 summary propositions (spec §5, human check protocol).

    # 1. after every generation run has finished: draw the blinded sample
    python scripts/kw_human_check.py make --config configs/kw_week1.toml

    # 2. annotators fill annotations/kw_human_check/labels_<name>.xlsx,
    #    reading annotations/kw_human_check/reading_pack.html
    #    -- or label on the web page: export the blinded items for it, and turn its labels
    #    (a JSON file {"annotators": {name: [per-summary documents]}}) back into those sheets
    python scripts/kw_human_check.py web-items --config configs/kw_week1.toml
    python scripts/kw_human_check.py from-web --config configs/kw_week1.toml --labels web_labels.json

    # 3. compare with the scorer (and with each other, when both sheets exist)
    python scripts/kw_human_check.py check --config configs/kw_week1.toml
    python scripts/kw_human_check.py check --config configs/kw_week1.toml --device cpu

The key that links items to model and condition is written to
results/kw/<run>/human_check_key.json, outside the annotation folder: annotators
must not open it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from fairsum.kwsum.collections import read_collections
from fairsum.kwsum.config import COLLECTIONS_PATH, REPO_ROOT, load_config
from fairsum.kwsum.human_check import (
    HumanCheckSheetError, binary_support, compare, item_ids, read_key, read_sheet, sample_summaries,
    scorer_labels, sentiment_view, support_view, unsupported_rate_gap, web_items, web_labels,
    write_key, write_reading_pack, write_sheet,
)
from fairsum.kwsum.store import read_jsonl

SHEET_DIR = REPO_ROOT / "annotations" / "kw_human_check"


def verdict(alpha: float) -> str:
    """Interpretation fixed in the spec before any label existed."""
    if alpha != alpha:  # NaN
        return "undefined"
    return "good" if alpha >= 0.80 else "acceptable" if alpha >= 0.667 else "EXPLORATORY ONLY"


def primary_faithfulness(alpha_as_written: float, alpha_deattr: float, margin: float = 0.05) -> str:
    """Registered 30 Sep 2026, before any human label: the de-attributed score becomes primary
    only if its supported-vs-not agreement with humans beats the registered score's by `margin`."""
    if alpha_deattr == alpha_deattr and alpha_as_written == alpha_as_written \
            and alpha_deattr - alpha_as_written >= margin:
        return "de-attributed"
    return "as written (registered)"


def make(args, cfg) -> None:
    sheets = [SHEET_DIR / f"labels_{name}.xlsx" for name in args.annotators]
    started = [p for p in sheets if p.exists()]
    if started and not args.force:
        raise SystemExit(f"refusing to overwrite existing sheets: {', '.join(map(str, started))} "
                         "(pass --force to start over)")
    gens = [g for g in read_jsonl(cfg.generations_path) if g.get("split", "test") == "test"]
    summaries = sample_summaries(gens, args.n_summaries, args.per_summary, cfg.seed,
                                 cfg.max_new_tokens)
    collections = {c.collection_id: c for c in read_collections(COLLECTIONS_PATH)}
    for path in sheets:
        write_sheet(summaries, path)
    write_reading_pack(summaries, collections, SHEET_DIR / "reading_pack.html")
    write_key(summaries, cfg.results_dir / "human_check_key.json", cfg.seed)
    models = sorted({s.model for s in summaries})
    print(f"{len(summaries)} summaries, {len(item_ids(summaries))} sentences, "
          f"models {models}, from {len(gens)} test generations")
    print(f"sheets: {', '.join(str(p) for p in sheets)}")
    print(f"reading pack: {SHEET_DIR / 'reading_pack.html'}")


def export_web_items(args, cfg) -> None:
    """The blinded items for the labelling page; the key stays in results/."""
    summaries = read_key(cfg.results_dir / "human_check_key.json")
    collections = {c.collection_id: c for c in read_collections(COLLECTIONS_PATH)}
    out = cfg.results_dir / "human_check_web_items.json"
    out.write_text(json.dumps({"items": web_items(summaries, collections)}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"{len(summaries)} summaries, {len(item_ids(summaries))} sentences -> {out}")


def import_web_labels(args, cfg) -> int:
    """One filled sheet per annotator from the page's labels, so `check` reads them unchanged."""
    summaries = read_key(cfg.results_dir / "human_check_key.json")
    data = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    for name, docs in data["annotators"].items():
        if not re.fullmatch(r"[a-z0-9-]{2,20}", name):
            print(f"skipped annotator name {name!r}: use 2-20 lowercase letters, digits or hyphens")
            continue
        path = SHEET_DIR / f"labels_{name}.xlsx"
        if path.exists() and not args.force:
            print(f"refusing to overwrite {path} (pass --force)")
            return 1
        labels = web_labels(docs)
        write_sheet(summaries, path, labels=labels)
        done = sum(1 for i in item_ids(summaries) if (labels.get(i) or {}).get("claim"))
        print(f"{name}: {done} of {len(item_ids(summaries))} items labelled -> {path}")
    return 0


def check(args, cfg) -> int:
    summaries = read_key(cfg.results_dir / "human_check_key.json")
    expected = item_ids(summaries)
    humans: dict[str, dict] = {}
    for name in args.annotators:
        path = SHEET_DIR / f"labels_{name}.xlsx"
        if not path.exists():
            print(f"(no sheet for {name} at {path}; skipped)")
            continue
        try:
            humans[name] = read_sheet(path, expected_ids=expected)
        except HumanCheckSheetError as err:
            print(f"{path}:\n{err}")
            return 1
    if not humans:
        print("no filled sheets found")
        return 1

    from fairsum.kwsum.scoring import HFNLI, build_sentiment

    collections = {c.collection_id: c for c in read_collections(COLLECTIONS_PATH)}
    scorer = scorer_labels(summaries, collections, HFNLI(cfg.nli_model, device=args.device),
                           build_sentiment(cfg.sentiment_model, cfg.neutral_model, args.device),
                           cfg.threshold)
    scorer_sent = {i: s["sentiment"] for i, s in scorer.items()}
    scorer_polar = {i: s["sentiment_polarity_only"] for i, s in scorer.items()}
    scorer_support = {i: s["support"] for i, s in scorer.items()}
    scorer_support_d = {i: s["support_deattr"] for i, s in scorer.items()}

    rows, report = [], {}
    for name, human in humans.items():
        n = len(human)
        claims = sum(l["claim"] == "yes" for l in human.values())
        mixed = sum(l["sentiment"] == "mixed" for l in human.values())
        sent = compare(sentiment_view(human), scorer_sent)
        polar_items = {i: v for i, v in sentiment_view(human).items() if v != "neutral"}
        polar = compare(polar_items, scorer_polar)
        human_sup = support_view(human)
        sup = compare(human_sup, scorer_support)
        sup_d = compare(human_sup, scorer_support_d)
        bin_reg = compare(binary_support(human_sup), binary_support(scorer_support))
        bin_d = compare(binary_support(human_sup), binary_support(scorer_support_d))
        print(f"\n=== {name} vs scorer ===")
        print(f"claims: {claims} of {n} ({n - claims} kept by the splitter but not claims)")
        print(f"mixed sentiment: {mixed} of {claims} claims (left out of the sentiment comparison)")
        for label, res in (("sentiment (hybrid, 3 classes)", sent),
                           ("sentiment (polarity model, human-polar items)", polar),
                           ("support as written (3 classes)", sup),
                           ("support de-attributed (3 classes)", sup_d),
                           ("support as written (supported vs not)", bin_reg),
                           ("support de-attributed (supported vs not)", bin_d)):
            print(f"{label}: n={res.n} accuracy={res.accuracy:.3f} alpha={res.alpha:.3f} "
                  f"-> {verdict(res.alpha)}")
            print("  confusion (human, scorer): "
                  + ", ".join(f"{h}->{s}: {c}" for (h, s), c in sorted(res.confusion.items())))
            rows.append({"comparison": f"{name} vs scorer", "label": label, "n": res.n,
                         "accuracy": res.accuracy, "alpha": res.alpha, "verdict": verdict(res.alpha)})
        gaps = {}
        for label, scorer_map in (("as written", scorer_support), ("de-attributed", scorer_support_d)):
            gap = gaps[label] = unsupported_rate_gap(human_sup, scorer_map)
            print(f"unsupported rate ({label}): human {gap['human_rate']:.3f}, scorer "
                  f"{gap['scorer_rate']:.3f}, scorer - human = {gap['diff']:+.3f} "
                  f"[{gap['ci_lo']:+.3f}, {gap['ci_hi']:+.3f}] ({gap['n_summaries']} summaries)")
        report[name] = {"n": n, "claims": claims, "mixed": mixed, "unsupported_gap": gaps,
                        "alpha_binary_as_written": bin_reg.alpha, "alpha_binary_deattr": bin_d.alpha}

    # The registered rule is applied once: on the items both annotators agree on, or on the
    # only annotator's labels.
    views = [support_view(h) for h in humans.values()]
    basis = views[0] if len(views) == 1 else {
        i: v for i, v in views[0].items() if i in views[1] and views[1][i] == v}
    a_reg = compare(binary_support(basis), binary_support(scorer_support)).alpha
    a_d = compare(binary_support(basis), binary_support(scorer_support_d)).alpha
    primary = primary_faithfulness(a_reg, a_d)
    print(f"\nprimary faithfulness score by the registered rule ({len(basis)} items"
          f"{', annotators agree' if len(views) > 1 else ''}): {primary} "
          f"(alpha as written {a_reg:.3f}, de-attributed {a_d:.3f})")
    report["primary_faithfulness"] = {"choice": primary, "n_items": len(basis),
                                      "alpha_as_written": a_reg, "alpha_deattr": a_d}

    names = list(humans)
    if len(names) >= 2:
        a, b = humans[names[0]], humans[names[1]]
        print(f"\n=== {names[0]} vs {names[1]} ===")
        for label, view in (("claim", lambda h: {i: l["claim"] for i, l in h.items()}),
                            ("sentiment (4 classes)",
                             lambda h: {i: l["sentiment"] for i, l in h.items() if l["claim"] == "yes"}),
                            ("support (3 classes)", support_view)):
            res = compare(view(a), view(b))
            print(f"{label}: n={res.n} accuracy={res.accuracy:.3f} alpha={res.alpha:.3f} "
                  f"-> {verdict(res.alpha)}")
            rows.append({"comparison": f"{names[0]} vs {names[1]}", "label": label, "n": res.n,
                         "accuracy": res.accuracy, "alpha": res.alpha, "verdict": verdict(res.alpha)})

    tables = cfg.results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "human_check.csv", index=False)
    (cfg.results_dir / "human_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {tables / 'human_check.csv'}, {cfg.results_dir / 'human_check.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["make", "check", "web-items", "from-web"])
    ap.add_argument("--labels", help="from-web: JSON file of the page's labels")
    ap.add_argument("--config", required=True)
    ap.add_argument("--annotators", nargs="+", default=["nishan", "prathyuman"])
    ap.add_argument("--n-summaries", type=int, default=25)
    ap.add_argument("--per-summary", type=int, default=4)
    ap.add_argument("--device", default=None, help="cpu or cuda for the scorer models (check)")
    ap.add_argument("--force", action="store_true", help="make, from-web: overwrite existing sheets")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.action == "make":
        make(args, cfg)
        return 0
    if args.action == "web-items":
        export_web_items(args, cfg)
        return 0
    if args.action == "from-web":
        return import_web_labels(args, cfg)
    return check(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
