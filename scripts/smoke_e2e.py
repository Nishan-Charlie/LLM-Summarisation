"""End-to-end smoke test: pool -> instances -> k samples -> verifier -> conformal.

Week 4's pipeline in miniature, on a handful of dev instances. Exists to prove
the pieces connect and to exercise the GPU paths that unit tests cannot.

The adverse events here are placeholders extracted by keyword, NOT the gold
annotations. Every number this prints is therefore meaningless as a result --
it only demonstrates that the plumbing runs.

Usage:
    python scripts/smoke_e2e.py --instances 6 --k 8
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fairsum.conformal import finite_sample_floor, fit_threshold, risk_coverage_curve
from fairsum.generation import BatchedGenerator, records_to_frame
from fairsum.instances import build_instances
from fairsum.prompts import PROMPT_BUILDERS
from fairsum.verifier import build_nli_verifier, coverage_loss, measure_coverage, select_best

# Placeholder stand-ins for gold annotation, so the pipeline has something to
# score. Replace with `fairsum.annotation` output once Week 2/3 delivers it.
PLACEHOLDER_EVENTS = {
    "dryness": ["dry", "dryness", "flaking", "peeling"],
    "irritation": ["irritat", "burning", "sting", "redness"],
    "nausea": ["nausea", "vomit", "sick to my stomach"],
    "drowsiness": ["drowsy", "sleepy", "tired", "fatigue", "zombie"],
    "dizziness": ["dizzy", "dizziness", "lightheaded"],
    "headache": ["headache", "migraine"],
    "weight gain": ["weight gain", "gained weight"],
    "pain": ["pain", "ache", "cramp"],
}


def placeholder_events(texts: list[str]) -> list[set[str]]:
    return [
        {
            event
            for event, cues in PLACEHOLDER_EVENTS.items()
            if any(cue in text.lower() for cue in cues)
        }
        for text in texts
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--nli", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--instances", type=int, default=6)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--method", default="direct", choices=sorted(PROMPT_BUILDERS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()

    print(">>> instances")
    pool = pd.read_csv("data/review_pool.csv")
    instances = build_instances(
        pool[pool["split"] == "dev"], n_reviews=8, seed=1, max_instances=args.instances
    )
    gold = {
        inst.instance_id: set().union(*placeholder_events(inst.texts))
        for inst in instances
    }
    print(f"    {len(instances)} instances, "
          f"mean |A| = {np.mean([len(g) for g in gold.values()]):.1f}")

    print(">>> generation")
    generator = BatchedGenerator(args.model)
    records = generator.generate_records(
        instance_ids=[i.instance_id for i in instances],
        prompts=[PROMPT_BUILDERS[args.method](i) for i in instances],
        method=args.method,
        k=args.k,
        batch_size=args.batch_size,
        max_new_tokens=320,
        seed=0,
    )
    frame = records_to_frame(records)
    print(f"    {len(frame)} generations, "
          f"parse ok {(frame['parse_status'] == 'ok').mean():.0%}, "
          f"{frame['completion_tokens'].sum()} completion tokens")

    print(">>> verifier")
    verify = build_nli_verifier(args.nli)

    scores, losses, abstain_counts = [], [], 0
    for inst in instances:
        events = sorted(gold[inst.instance_id])
        candidates = [
            r.summary for r in records
            if r.instance_id == inst.instance_id and r.summary
        ]
        if not candidates or not events:
            continue
        reports = [measure_coverage(c, events, verify) for c in candidates]
        cand_scores = [1.0 - coverage_loss(r) for r in reports]
        abstain_counts += sum(len(r.abstained) for r in reports)

        _, best_score, best_idx = select_best(candidates, cand_scores)
        scores.append(best_score)
        losses.append(coverage_loss(reports[best_idx]))

    print(f"    scored {len(scores)} instances, {abstain_counts} verifier abstentions")
    if not scores:
        print("    nothing scored -- no parseable summaries or no events found")
        return 1

    print(">>> conformal")
    scores_arr, losses_arr = np.array(scores), np.array(losses)
    floor = finite_sample_floor(len(scores_arr))
    print(f"    n={len(scores_arr)}, finite-sample floor={floor:.3f}")
    alpha = max(args.alpha, floor * 1.5)
    if alpha != args.alpha:
        print(f"    raised alpha {args.alpha} -> {alpha:.3f} (n far too small for the ask)")

    result = fit_threshold(scores_arr, losses_arr, alpha=alpha)
    print(f"    {result}")
    print("\n" + risk_coverage_curve(scores_arr, losses_arr, n_points=6)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nPipeline ran end to end. The NUMBERS ARE MEANINGLESS -- placeholder "
          "keyword events, n far below what any guarantee needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
