"""Part B of the human evaluation: whole summaries judged by people (amendments 9 and 10).

    # 1. draw the registered sample (80 sets) and write the key and the page items
    python scripts/kw_human_partb.py make --config configs/kw_week1.toml

    # 2. the labelling site's data, with who labels which sets (two pairs, or two people for all 80)
    python scripts/kw_human_partb.py site-data --config configs/kw_week1.toml --pair1 ann1 ann2 --pair2 ann3 ann4
    python scripts/kw_human_partb.py site-data --config configs/kw_week1.toml --all ann1 ann2
    #    then load tools/kw_label_site/data/partb.json through PUT /api/b/seed

    # 3. after labelling: GET /api/b/export into a file, then
    python scripts/kw_human_partb.py check --config configs/kw_week1.toml --labels partb_export.json

The key that links sets to collection, model and prompt order is written to
results/kw/<run>/human_partb_key.json; it is never deployed. Analysis details:
docs/design/2026-10-30-kw-partb-analysis.md.
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from fairsum.kwsum.collections import read_collections
from fairsum.kwsum.config import COLLECTIONS_PATH, REPO_ROOT, load_config
from fairsum.kwsum.human_check import read_key as read_part_a_key
from fairsum.kwsum.human_partb import (
    GUIDE, PRACTICE_COLLECTION, SEED, analyse, judgments, practice_item, read_key, sample_sets,
    web_items, web_labels, write_key,
)
from fairsum.kwsum.scoring import SCORER_VERSION
from fairsum.kwsum.store import read_jsonl

SITE_DATA = REPO_ROOT / "tools" / "kw_label_site" / "data" / "partb.json"
NAME = re.compile(r"^[a-z0-9-]{2,20}$")
PRACTICE_MODEL = "llama3.2-3b"


def key_path(cfg):
    return cfg.results_dir / "human_partb_key.json"


def make(args, cfg) -> None:
    if key_path(cfg).exists() and not args.force:
        raise SystemExit(f"refusing to redraw: {key_path(cfg)} exists (pass --force to start over)")
    collections = {c.collection_id: c for c in read_collections(COLLECTIONS_PATH)}
    part_a = {s.collection_id for s in read_part_a_key(cfg.results_dir / "human_check_key.json")}
    sets = sample_sets(read_jsonl(cfg.scores_path), read_jsonl(cfg.generations_path), collections,
                       exclude=part_a, models=[m.name for m in cfg.models], seed=SEED)
    write_key(sets, key_path(cfg), SEED)
    items = web_items(sets, collections)
    (cfg.results_dir / "human_partb_web_items.json").write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(sets)} sets from {len({s.model for s in sets})} models, "
          f"{sum(s.a_condition == 'direct' for s in sets)} with the plain summary as A; "
          f"{len(part_a)} Part A collections and the practice collection excluded")
    print(f"key: {key_path(cfg)}\npage items: {cfg.results_dir / 'human_partb_web_items.json'}")


def site_data(args, cfg) -> None:
    if bool(args.all) == bool(args.pair1 or args.pair2):
        raise SystemExit("give either --all A B, or --pair1 A B and --pair2 C D")
    if args.pair1 or args.pair2:
        if not (args.pair1 and args.pair2):
            raise SystemExit("--pair1 and --pair2 go together")
        assign = {**{n: ["g1", "shared"] for n in args.pair1}, **{n: ["g2", "shared"] for n in args.pair2}}
    else:
        assign = {n: ["g1", "g2", "shared"] for n in args.all}
    bad = [n for n in assign if not NAME.match(n)]
    if bad or len(assign) != len(args.all or []) + len(args.pair1 or []) + len(args.pair2 or []):
        raise SystemExit(f"annotator names must be distinct, 2-20 lowercase letters, digits or hyphens: {bad}")
    items = json.loads((cfg.results_dir / "human_partb_web_items.json").read_text(encoding="utf-8"))["items"]
    collections = {c.collection_id: c for c in read_collections(COLLECTIONS_PATH)}
    texts = {g["condition"]: g["raw"] for g in read_jsonl(cfg.generations_path)
             if g["collection_id"] == PRACTICE_COLLECTION and g["model"] == PRACTICE_MODEL}
    practice = practice_item(collections[PRACTICE_COLLECTION], texts["direct"], texts["kw_strat"])
    data = {"status": args.status, "guide": GUIDE, "assign": assign, "items": [practice, *items]}
    SITE_DATA.parent.mkdir(exist_ok=True)
    SITE_DATA.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    sizes = {n: sum(it.get("group") in g for it in items) for n, g in assign.items()}
    print(f"{len(items)} sets + practice, status {args.status}; sets per annotator: {sizes}\n-> {SITE_DATA}")


def check(args, cfg) -> int:
    export = json.loads(open(args.labels, encoding="utf-8").read())
    sets = read_key(key_path(cfg))
    labels = {who: web_labels(docs) for who, docs in export.get("annotators", {}).items()}
    scores = {(r["collection_id"], r["model"], r["condition"]): r for r in read_jsonl(cfg.scores_path)
              if r.get("scorer_version") == SCORER_VERSION}
    part_a_path = cfg.results_dir / "human_check.json"
    part_a = json.loads(part_a_path.read_text(encoding="utf-8")) if part_a_path.exists() else None
    report = analyse(sets, labels, scores, assign=export.get("assign"), part_a=part_a)
    report["exported_at"] = export.get("exported_at")

    print(f"annotators analysed: {report['annotators']}; left out (unfinished): {report['excluded']}")
    if "q1" in report:
        q1 = report["q1"]
        print(f"Q1 coverage, kw_strat - direct: {q1['mean_diff']:+.3f} [{q1['ci_lo']:+.3f}, {q1['ci_hi']:+.3f}] "
              f"on {q1['n_sets']} sets, Wilcoxon p = {q1['p_wilcoxon']:.4g}; "
              f"Spearman with minority recall {q1['spearman']['registered']['rho']}")
        s = report["q2"]["sign_test"]
        print(f"Q2 sign test: kw_strat more balanced in {s['kw_strat_more_balanced']}, direct in "
              f"{s['direct_more_balanced']}, ties {s['ties']}, p = {s['p']}")
        for q, a in report["agreement"].items():
            print(f"alpha {q} ({a['metric']}): {a['alpha']} [{a['ci_lo']}, {a['ci_hi']}]")
    print("checks:", json.dumps(report["checks"]))

    tables = cfg.results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    use = {who: labels[who] for who in report["annotators"]}
    pd.DataFrame(judgments(sets, use, scores)).to_csv(tables / "human_partb_judgments.csv", index=False)
    out = cfg.results_dir / "human_partb.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written: {out}, {tables / 'human_partb_judgments.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["make", "site-data", "check"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", help="check: the site's Part B export")
    ap.add_argument("--pair1", nargs=2, metavar="NAME", help="site-data: the pair that labels group 1")
    ap.add_argument("--pair2", nargs=2, metavar="NAME", help="site-data: the pair that labels group 2")
    ap.add_argument("--all", nargs=2, metavar="NAME", help="site-data: two annotators who label all 80")
    ap.add_argument("--status", choices=["practice", "open", "closed"], default="open")
    ap.add_argument("--force", action="store_true", help="make: redraw over an existing key")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.action == "make":
        make(args, cfg)
        return 0
    if args.action == "site-data":
        site_data(args, cfg)
        return 0
    if not args.labels:
        raise SystemExit("check needs --labels")
    return check(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
