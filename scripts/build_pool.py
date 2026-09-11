"""Days 3 and 5 -- build the review pool, assign drug splits, emit an annotation batch.

Splits are drug-disjoint and assigned once under a recorded seed. Nothing here
looks at the test split again; that is the point of writing the manifest now.

Usage:
    python scripts/build_pool.py --seed 42
    python scripts/build_pool.py --seed 42 --annotation-batch 50 --annotation-drugs 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fairsum.data import build_review_pool, load_druglib_raw
from fairsum.instances import assign_drug_splits, build_instances


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-reviews", type=int, default=8)
    parser.add_argument("--annotation-batch", type=int, default=0,
                        help="emit this many reviews for the Day 3 pilot")
    parser.add_argument("--annotation-drugs", type=int, default=15,
                        help="spread the pilot batch across at least this many drugs")
    parser.add_argument("--group-by", default="drug",
                        choices=("drug", "drug_indication"),
                        help="drug_indication is stricter but costs ~70%% of instances")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading Druglib ...")
    pool = build_review_pool(load_druglib_raw(args.out))
    pool.to_csv(args.out / "review_pool.csv", index=False)
    print(f"  reviews {len(pool)}, drugs {pool['product_id'].nunique()}, "
          f"(drug, indication) groups {pool.groupby(['product_id','condition_name']).ngroups}")

    splits = assign_drug_splits(pool["product_id"].unique(), seed=args.seed)
    pool["split"] = pool["product_id"].map(splits)
    pool.to_csv(args.out / "review_pool.csv", index=False)

    manifest = {
        "seed": args.seed,
        "n_reviews_per_instance": args.n_reviews,
        "splits": splits,
        "counts": pool["split"].value_counts().to_dict(),
        "drugs_per_split": pool.groupby("split")["product_id"].nunique().to_dict(),
    }
    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nDrug-disjoint splits:")
    print(pool.groupby("split").agg(
        reviews=("review_id", "size"), drugs=("product_id", "nunique")
    ).to_string())

    # How many instances the pool can actually yield, per split.
    print(f"\nAvailable instances (group_by={args.group_by}):")
    rows = []
    for split in ("dev", "calibration", "test"):
        drugs = {d for d, s in splits.items() if s == split}
        rows.append({
            "split": split,
            "drugs_in_split": len(drugs),
            "usable_drugs": len(build_instances(
                pool, n_reviews=args.n_reviews, seed=args.seed,
                drugs=drugs, group_by=args.group_by,
            )),
            "instances_5_draws": len(build_instances(
                pool, n_reviews=args.n_reviews, seed=args.seed, drugs=drugs,
                group_by=args.group_by, instances_per_group=5,
            )),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nNOTE: `instances_5_draws` reuses reviews, so it inflates instance count "
          "without adding drugs. Effective n is closer to the drug column -- that is "
          "the Day 5 question (`Calibration Design`).")

    if args.annotation_batch:
        rng = np.random.default_rng(args.seed)
        candidates = pool[pool["split"] != "test"]
        drugs = list(candidates["product_id"].unique())
        rng.shuffle(drugs)
        chosen = drugs[: max(args.annotation_drugs, 1)]

        # Round-robin across drugs so the batch reaches the requested size even
        # when some drugs have few reviews, and stays spread rather than
        # concentrating on whichever drugs happen to be large.
        by_drug = {
            drug: group.sample(frac=1.0, random_state=args.seed).to_dict("records")
            for drug, group in candidates[
                candidates["product_id"].isin(chosen)
            ].groupby("product_id")
        }
        picked_rows = []
        while len(picked_rows) < args.annotation_batch and any(by_drug.values()):
            for drug in list(by_drug):
                if not by_drug[drug]:
                    continue
                picked_rows.append(by_drug[drug].pop())
                if len(picked_rows) >= args.annotation_batch:
                    break
        picked = pd.DataFrame(picked_rows)

        batch_path = args.out / "annotation_batch_pilot.csv"
        picked[["review_id", "product_id", "condition_name", "text"]].to_csv(
            batch_path, index=False
        )
        print(f"\nAnnotation pilot batch: {len(picked)} reviews across "
              f"{picked['product_id'].nunique()} drugs -> {batch_path}")
        print("Annotate with fairsum.annotation.AdverseEvent "
              "(span, event, certainty, attribution), save as JSONL.")
        print("Watch for the two failure modes: negation and condition-attribution.")

    print(f"\nManifest: {args.out / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
