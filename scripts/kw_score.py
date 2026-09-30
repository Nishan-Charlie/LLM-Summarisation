"""Score generations (spec §5). Resumable; uses the GPU for NLI and sentiment.

    python scripts/kw_score.py --config configs/kw_week1.toml
    python scripts/kw_score.py --config configs/kw_week1.toml --model llama3.2-3b
    python scripts/kw_score.py --config configs/kw_week1.toml --validate-sentiment
"""

from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from fairsum.kwsum.collections import read_collections
from fairsum.kwsum.config import COLLECTIONS_PATH, KEYWORDS_PATH, load_config
from fairsum.kwsum.manifest import check_or_write_manifest, input_identity
from fairsum.kwsum.scoring import (
    SCORER_VERSION, HFNLI, score_generation, validate_sentiment,
)
from fairsum.kwsum.scoring import build_sentiment as _build_sentiment
from fairsum.kwsum.store import append_jsonl, read_jsonl

KEY = ("collection_id", "condition", "model")


def build_sentiment(cfg):
    """siebert for polarity; a 3-class neutral gate in front when configured."""
    return _build_sentiment(cfg.sentiment_model, cfg.neutral_model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="", help="score only this model's generations")
    ap.add_argument("--validate-sentiment", action="store_true")
    ap.add_argument("--flush-every", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    collections = {c.collection_id: c for c in read_collections(COLLECTIONS_PATH)}

    if args.validate_sentiment:
        # Validate the instrument actually used for scoring (hybrid if configured).
        report = validate_sentiment(build_sentiment(cfg), list(collections.values()))
        report.update(sentiment_model=cfg.sentiment_model, neutral_model=cfg.neutral_model)
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        (cfg.results_dir / "sentiment_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    check_or_write_manifest(cfg.results_dir / "run_manifest.json",
                            input_identity(COLLECTIONS_PATH, KEYWORDS_PATH))
    keywords = {row["collection_id"]: row for row in read_jsonl(KEYWORDS_PATH)}
    # Resume only over rows scored by this scorer version; older rows are ignored here
    # and dropped by the analysis.
    scored = read_jsonl(cfg.scores_path)
    done = {tuple(r[k] for k in KEY) for r in scored if r.get("scorer_version") == SCORER_VERSION}
    stale = sum(r.get("scorer_version") != SCORER_VERSION for r in scored)
    todo = [g for g in read_jsonl(cfg.generations_path)
            if (not args.model or g["model"] == args.model)
            and tuple(g[k] for k in KEY) not in done]
    print(f"{len(todo)} generations to score; {len(done)} already scored by v{SCORER_VERSION}; "
          f"{stale} stale rows from other versions ignored")
    if not todo:
        return

    nli, sentiment = HFNLI(cfg.nli_model), build_sentiment(cfg)
    buffer: list[dict] = []
    for g in tqdm(todo, desc="scoring"):
        cid = g["collection_id"]
        buffer.append(score_generation(g, collections[cid], keywords[cid], nli, sentiment,
                                       cfg.threshold, max_new_tokens=cfg.max_new_tokens))
        if len(buffer) >= args.flush_every:
            append_jsonl(cfg.scores_path, buffer)
            buffer = []
    append_jsonl(cfg.scores_path, buffer)


if __name__ == "__main__":
    main()
