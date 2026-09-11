"""Day 4 -- the schema gate.

    alpha >= 0.7  freeze the schema, start bulk annotation
    0.6 - 0.7     one revision, re-pilot on 30 fresh reviews
    alpha <  0.6  stop; the construct is not annotatable as defined

Usage:
    python scripts/check_agreement.py --a annotations/nishan.jsonl --b annotations/prathyuman.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fairsum.annotation import agreement_report, load_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="annotator A JSONL")
    parser.add_argument("--b", type=Path, required=True, help="annotator B JSONL")
    args = parser.parse_args()

    events_a = {a.review_id: a.event_set for a in load_annotations(args.a)}
    events_b = {a.review_id: a.event_set for a in load_annotations(args.b)}

    report = agreement_report(events_a, events_b)
    print(report.summary())

    if report.disagreements:
        print("\nDisagreements to adjudicate:")
        for review_id, event in report.disagreements:
            in_a = "A" if event in events_a.get(review_id, set()) else " "
            in_b = "B" if event in events_b.get(review_id, set()) else " "
            print(f"  [{in_a}{in_b}] {review_id}  {event}")
        print("\nClassify each: genuine ambiguity, schema gap, or annotator slip. "
              "Only schema gaps justify revising the schema.")

    print("\n" + {
        "freeze": "GATE PASSED -- freeze the schema and start bulk annotation.",
        "revise": "BORDERLINE -- one schema revision, then re-pilot on 30 fresh reviews.",
        "stop": "GATE FAILED -- stop. Rethink the construct before annotating further.",
        "undefined": "UNDEFINED -- not enough rating variance to compute alpha. "
                     "Widen the pilot sample.",
    }[report.verdict])
    return 0 if report.verdict in ("freeze", "revise") else 1


if __name__ == "__main__":
    raise SystemExit(main())
