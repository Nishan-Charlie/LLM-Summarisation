"""Days 3-4 -- a spreadsheet to annotate, and the JSONL that agreement reads.

Usage:
    python scripts/annotation_sheet.py export --batch data/annotation_batch_pilot.csv --out annotations/nishan.xlsx
    python scripts/annotation_sheet.py export --batch data/annotation_batch_pilot.csv --out annotations/prathyuman.xlsx --subset 30
    python scripts/annotation_sheet.py import --sheet annotations/nishan.xlsx --out annotations/nishan.jsonl

One row per review: read the text, fill an event slot per adverse event, mark the
row done. A review with no adverse events is still marked done.

`import` refuses a sheet with any review not marked done, a partly filled slot,
or a blank certainty or attribution, and lists every problem at once. The second
annotator's sheet carries no labels, so the two passes stay independent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fairsum.annotation import AnnotationSheetError, save_annotations
from fairsum.annotation_sheet import export_sheet, load_sheet_annotations
from fairsum.instances import select_agreement_subset


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="batch CSV -> sheet to annotate")
    export.add_argument("--batch", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--subset", type=int,
                        help="export only this many reviews, spread across drugs")
    export.add_argument("--seed", type=int, default=42)

    load = commands.add_parser("import", help="filled sheet -> annotations JSONL")
    load.add_argument("--sheet", type=Path, required=True)
    load.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "export":
        batch = pd.read_csv(args.batch)
        if args.subset:
            batch = select_agreement_subset(batch, size=args.subset, seed=args.seed)
        export_sheet(batch, args.out)
        print(f"Wrote {len(batch)} reviews across {batch['product_id'].nunique()} "
              f"drugs -> {args.out}")
        return 0

    try:
        annotations = load_sheet_annotations(args.sheet)
    except AnnotationSheetError as err:
        print(err, file=sys.stderr)
        return 1
    save_annotations(annotations, args.out)
    n_events = sum(len(a.events) for a in annotations)
    print(f"Wrote {len(annotations)} reviews, {n_events} events -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
