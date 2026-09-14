"""Day 4 -- the schema gate.

    alpha >= 0.7  freeze the schema, start bulk annotation
    0.6 - 0.7     one revision, re-pilot on 30 fresh reviews
    alpha <  0.6  stop; the construct is not annotatable as defined

Usage:
    python scripts/check_agreement.py --a annotations/nishan.jsonl --b annotations/prathyuman.jsonl
    python scripts/check_agreement.py --a ... --b ... --synonyms annotations/synonyms.csv

Labels are compared as exact strings. `--synonyms` merges surface forms through
a `surface,canonical` table built during adjudication; agreement is printed
both before and after, because merging can only remove disagreements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fairsum.annotation import (
    agreement_report,
    canonicalise,
    load_annotations,
    load_synonym_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="annotator A JSONL")
    parser.add_argument("--b", type=Path, required=True, help="annotator B JSONL")
    parser.add_argument("--synonyms", type=Path, help="surface,canonical CSV")
    args = parser.parse_args()

    loaded_a = {a.review_id: a.event_set for a in load_annotations(args.a)}
    loaded_b = {a.review_id: a.event_set for a in load_annotations(args.b)}

    # Case and whitespace are never disagreements, so "as written" still folds them.
    events_a, events_b = canonicalise(loaded_a, {}), canonicalise(loaded_b, {})
    report = agreement_report(events_a, events_b)
    print("=== labels as written ===")
    print(report.summary())

    if args.synonyms:
        table = load_synonym_table(args.synonyms)
        before = report.n_disagreements
        events_a, events_b = canonicalise(loaded_a, table), canonicalise(loaded_b, table)
        report = agreement_report(events_a, events_b)
        print(f"\n=== after the synonym table ({len(table)} mappings) ===")
        print(report.summary())
        print(f"\nThe table removed {before - report.n_disagreements} of {before} "
              "disagreements. Report both results; the gate below uses merged labels.")

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
