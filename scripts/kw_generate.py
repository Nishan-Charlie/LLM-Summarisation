"""Generate summaries for every (collection, condition) with one model.

    python scripts/kw_generate.py --config configs/kw_week1.toml --model llama3.2-3b
    python scripts/kw_generate.py --config configs/kw_week1.toml --model llama3.2-3b \
        --datasets amazon --split test --limit-per-cell 4                     # pilot

Resumable. Rows already in results/kw/<run>/generations.jsonl for this model
are skipped, and each chunk is appended as soon as it finishes, so an
interrupted overnight run loses at most one chunk. Prompts run shortest-first
to keep padding low inside a batch.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from fairsum.kwsum.backends import DecodingConfig, build_backend
from fairsum.kwsum.collections import Collection, read_collections
from fairsum.kwsum.compliance import classify_output
from fairsum.kwsum.config import COLLECTIONS_PATH, KEYWORDS_PATH, load_config
from fairsum.kwsum.manifest import check_or_write_manifest, environment_info, input_identity
from fairsum.kwsum.prompts import build_prompt
from fairsum.kwsum.store import append_jsonl, done_keys, read_jsonl

KEY = ("collection_id", "condition", "model")


def select(collections: list[Collection], datasets, levels, split: str,
           limit_per_cell: int | None) -> list[Collection]:
    out, seen = [], {}
    for c in collections:
        if datasets and c.dataset not in datasets:
            continue
        if levels and c.level not in levels:
            continue
        if split != "all" and c.split != split:
            continue
        cell = (c.dataset, c.split, c.level)
        if limit_per_cell is not None and seen.get(cell, 0) >= limit_per_cell:
            continue
        seen[cell] = seen.get(cell, 0) + 1
        out.append(c)
    return out


def peak_vram_gb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["test", "dev", "all"], default="all")
    ap.add_argument("--datasets", default="", help="comma-separated; default all")
    ap.add_argument("--levels", default="", help="comma-separated; default all")
    ap.add_argument("--conditions", default="", help="comma-separated; default the config's")
    ap.add_argument("--limit-per-cell", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the configured batch size (the queue retries with 4, then 2, "
                         "after a GPU failure on long prompts)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.model(args.model)
    # Refuse to append to a run whose collections or keywords have since changed.
    check_or_write_manifest(cfg.results_dir / "run_manifest.json",
                            input_identity(COLLECTIONS_PATH, KEYWORDS_PATH),
                            info={**environment_info(), "max_new_tokens": cfg.max_new_tokens,
                                  "repetition_penalty": cfg.repetition_penalty})
    pick = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    collections = select(read_collections(COLLECTIONS_PATH), pick(args.datasets),
                         pick(args.levels), args.split, args.limit_per_cell)
    keywords = {row["collection_id"]: row for row in read_jsonl(KEYWORDS_PATH)}
    conditions = pick(args.conditions) or list(cfg.conditions)

    done = done_keys(cfg.generations_path, KEY)
    todo = [(c, cond) for c in collections for cond in conditions
            if (c.collection_id, cond, spec.name) not in done]
    prompts = {(c.collection_id, cond): build_prompt(cond, c, keywords[c.collection_id])
               for c, cond in todo}
    todo.sort(key=lambda item: len(prompts[(item[0].collection_id, item[1])]))
    print(f"{spec.name}: {len(todo)} generations to run; {len(done)} rows already stored")
    if not todo:
        return

    decoding = DecodingConfig(cfg.max_new_tokens, cfg.repetition_penalty,
                              args.batch_size or spec.batch_size or cfg.batch_size)
    backend = build_backend(spec, decoding)
    finished, started = 0, time.time()
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start : start + args.chunk]
        t0 = time.time()
        completions = backend.complete([prompts[(c.collection_id, cond)] for c, cond in chunk])
        per_item = (time.time() - t0) / len(chunk)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        append_jsonl(cfg.generations_path, [
            {
                "collection_id": c.collection_id, "dataset": c.dataset, "split": c.split,
                "level": c.level, "condition": cond, "model": spec.name,
                "model_id": spec.model_id, "raw": comp.text,
                "status": classify_output(comp.text, comp.completion_tokens, cfg.max_new_tokens,
                                          stop_reason=comp.stop_reason),
                "stop_reason": comp.stop_reason,
                "prompt_tokens": comp.prompt_tokens,
                "completion_tokens": comp.completion_tokens,
                "seconds_per_item": round(per_item, 3), "created_at": stamp,
            }
            for (c, cond), comp in zip(chunk, completions)
        ])
        finished += len(chunk)
        rate = finished / (time.time() - started)
        eta_h = (len(todo) - finished) / rate / 3600
        print(f"  {finished}/{len(todo)}  {rate:.2f} gen/s  eta {eta_h:.1f} h  "
              f"peak VRAM {peak_vram_gb()} GB", flush=True)


if __name__ == "__main__":
    main()
