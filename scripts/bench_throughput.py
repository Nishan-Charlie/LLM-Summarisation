"""Day 2 -- measure real generation throughput on this machine.

v1's sequential loop managed 540 short generations in 50 minutes on a 1B model.
The SLMs programme measured this same laptop (RTX 4070 Laptop, 8 GB) at 2.22
chains/s at batch 32, so the projection is roughly a 30-100x speedup from
batching alone. This script replaces the projection with a measurement.

Usage:
    python scripts/bench_throughput.py --model meta-llama/Llama-3.2-1B-Instruct
    python scripts/bench_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --4bit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from fairsum.generation import BatchedGenerator
from fairsum.instances import Instance
from fairsum.prompts import PROMPT_BUILDERS, parse_final_summary, requires_marker


def dummy_instance(n_reviews: int = 8, words: int = 110) -> Instance:
    body = " ".join(["the medication helped somewhat but caused side effects"] * (words // 8))
    return Instance(
        instance_id="bench",
        product_id="benchdrug",
        condition_name="chronic pain",
        reviews=[
            {"review_id": f"r{i}", "text": f"{body} (review {i})"} for i in range(n_reviews)
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--method", default="ssfp", choices=sorted(PROMPT_BUILDERS))
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    args = parser.parse_args()

    import torch

    print(f"model      : {args.model}")
    print(f"quantized  : {args.four_bit}")
    print(f"cuda       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device     : {torch.cuda.get_device_name(0)}")

    generator = BatchedGenerator(args.model, load_in_4bit=args.four_bit)
    prompt = PROMPT_BUILDERS[args.method](dummy_instance())
    prompts = [prompt] * args.prompts

    rows = []
    for batch_size in args.batches:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        outputs = generator.generate(
            prompts,
            k=1,
            batch_size=batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.perf_counter() - start

        flat = [sample for group in outputs for sample in group]
        completion_tokens = sum(n_completion for _, _, n_completion in flat)
        needs_marker = requires_marker(args.method)
        parsed_ok = sum(
            parse_final_summary(raw, requires_marker=needs_marker).status.value == "ok"
            for raw, _, _ in flat
        )
        peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        )

        rows.append(
            {
                "batch": batch_size,
                "gen_per_s": len(flat) / elapsed,
                "tok_per_s": completion_tokens / elapsed,
                "peak_GB": peak_gb,
                "marker_ok_rate": parsed_ok / len(flat),
                "proj_8000_gens_h": (8000 / (len(flat) / elapsed)) / 3600,
            }
        )
        print(f"  batch {batch_size:>3}: {rows[-1]['gen_per_s']:.2f} gen/s, "
              f"peak {peak_gb:.2f} GB, marker ok {rows[-1]['marker_ok_rate']:.0%}")

    frame = pd.DataFrame(rows)
    print("\n" + frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nproj_8000_gens_h = 500 instances x k=16, one method.")
    print("marker_ok_rate below ~0.9 means max-new-tokens is too tight for this prompt.")

    out = Path("results") / f"day02_throughput_{args.model.split('/')[-1]}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
