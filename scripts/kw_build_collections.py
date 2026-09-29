"""Build review pools, imbalanced collections and keyword sets (spec §3-4).

    python scripts/kw_build_collections.py --config configs/kw_week1.toml

Pools are cached as parquet under data/kw/pools/ (large, regenerable,
gitignored). Collections and keywords are rewritten on every run so they always
match the config. The raw JSONL comes from the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from tqdm import tqdm

from fairsum.kwsum.collections import LEVELS, build_collections, write_collections
from fairsum.kwsum.config import (
    COLLECTIONS_PATH, KEYWORDS_PATH, MANIFEST_PATH, POOLS_DIR, load_config,
)
from fairsum.kwsum.keywords import collection_keywords, sentence_transformer_embedder
from fairsum.kwsum.sources import POOL_BUILDERS
from fairsum.kwsum.store import append_jsonl


def load_or_build_pool(dataset: str, rebuild: bool) -> pd.DataFrame:
    path = POOLS_DIR / f"{dataset}.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    start = time.time()
    pool = POOL_BUILDERS[dataset]()
    path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(path, index=False)
    print(f"[{dataset}] pool built in {time.time() - start:.0f}s")
    return pool


def pool_stats(pool: pd.DataFrame) -> dict:
    counts = pool.groupby(["entity", "polarity"]).size().unstack(fill_value=0)
    eligible = {
        level: int(((counts.get("positive", 0) >= p) & (counts.get("negative", 0) >= n)).sum())
        for level, (p, n) in LEVELS.items()
    }
    return {
        "reviews": int(len(pool)),
        "entities": int(pool["entity"].nunique()),
        "positive_share": round(float((pool["polarity"] == "positive").mean()), 4),
        "eligible_entities_by_level": eligible,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rebuild-pools", action="store_true")
    ap.add_argument("--skip-keywords", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    collections, stats = [], {}
    for dataset in cfg.datasets:
        pool = load_or_build_pool(dataset, args.rebuild_pools)
        stats[dataset] = pool_stats(pool)
        print(f"[{dataset}] {json.dumps(stats[dataset])}")
        built = build_collections(pool, dataset, cfg.n_test, cfg.n_dev, cfg.levels,
                                  cfg.dev_fraction, cfg.seed)
        collections.extend(built)
        print(f"[{dataset}] {len(built)} collections")

    write_collections(collections, COLLECTIONS_PATH)
    MANIFEST_PATH.write_text(json.dumps({
        "seed": cfg.seed, "n_test": cfg.n_test, "n_dev": cfg.n_dev,
        "levels": list(cfg.levels), "dev_fraction": cfg.dev_fraction,
        "n_collections": len(collections), "pools": stats,
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(collections)} collections to {COLLECTIONS_PATH}")

    if args.skip_keywords:
        return
    embed = sentence_transformer_embedder(cfg.embedder)
    KEYWORDS_PATH.unlink(missing_ok=True)
    rows = [collection_keywords(c, embed, k=cfg.k, diversity=cfg.diversity)
            for c in tqdm(collections, desc="keywords")]
    append_jsonl(KEYWORDS_PATH, rows)
    print(f"wrote keywords to {KEYWORDS_PATH}")


if __name__ == "__main__":
    main()
