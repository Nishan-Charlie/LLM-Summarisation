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

import pandas as pd

from fairsum.data import build_review_pool, load_druglib_raw
from fairsum.instances import assign_drug_splits, build_instances, select_annotation_batch


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
        # Only drugs that can form an instance, spread evenly. The first pilot
        # took any drug, and half its reviews could never be used downstream.
        picked = select_annotation_batch(
            pool,
            batch_size=args.annotation_batch,
            n_drugs=args.annotation_drugs,
            n_reviews=args.n_reviews,
            seed=args.seed,
        )

        batch_path = args.out / "annotation_batch_pilot.csv"
        picked[["review_id", "product_id", "condition_name", "text"]].to_csv(
            batch_path, index=False
        )
        per_drug = picked["product_id"].value_counts()
        print(f"\nAnnotation pilot batch: {len(picked)} reviews across "
              f"{len(per_drug)} drugs ({per_drug.min()}-{per_drug.max()} each) -> {batch_path}")
        print("Export a sheet to annotate: python scripts/annotation_sheet.py export "
              f"--batch {batch_path} --out annotations/<name>.xlsx")
        print("Watch for the two failure modes: negation and condition-attribution.")

    print(f"\nManifest: {args.out / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
