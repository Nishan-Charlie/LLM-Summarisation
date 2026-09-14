"""Gold adverse-event annotation: schema, storage, and agreement.

The critical path. Everything downstream -- the coverage loss, the conformal
calibration, every hypothesis -- is void without these labels, which is why the
Week 1 plan pilots the schema on 50 reviews before committing to 720.

`certainty` and `attribution` exist because the two predictable failure modes
are negation ("no side effects at all") and misattribution ("the pain was from
the fibromyalgia, not the drug"). A naive extractor makes both mistakes, and
both would silently inflate the gold event set.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Day 4 gates from the Week 1 plan.
FREEZE_THRESHOLD = 0.7
STOP_THRESHOLD = 0.6


class Certainty(Enum):
    ASSERTED = "asserted"
    HEDGED = "hedged"
    NEGATED = "negated"


class Attribution(Enum):
    DRUG = "drug"
    CONDITION = "condition"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class AdverseEvent:
    span: str
    event: str
    certainty: Certainty = Certainty.ASSERTED
    attribution: Attribution = Attribution.DRUG

    @property
    def counts_as_adverse_event(self) -> bool:
        """Negated events and those blamed on the condition are not adverse events.

        Hedged events do count: "might have made me drowsy" is still a reported
        adverse experience, and excluding it would systematically drop the
        tentative language patients use for the effects they are least sure of.
        """
        return (
            self.certainty is not Certainty.NEGATED
            and self.attribution is not Attribution.CONDITION
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["certainty"] = self.certainty.value
        d["attribution"] = self.attribution.value
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> AdverseEvent:
        return cls(
            span=d["span"],
            event=d["event"],
            certainty=Certainty(d.get("certainty", "asserted")),
            attribution=Attribution(d.get("attribution", "drug")),
        )


@dataclass(frozen=True)
class ReviewAnnotation:
    review_id: str
    events: list[AdverseEvent] = field(default_factory=list)

    @property
    def event_set(self) -> set[str]:
        """Canonical labels of the events that count, deduplicated."""
        return {e.event for e in self.events if e.counts_as_adverse_event}

    def to_dict(self) -> dict:
        return {
            "review_id": self.review_id,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> ReviewAnnotation:
        return cls(
            review_id=d["review_id"],
            events=[AdverseEvent.from_dict(e) for e in d.get("events", [])],
        )


def normalise_event(surface: str, table: Mapping[str, str]) -> str:
    """Map a surface form to its canonical label via the normalisation table.

    The table is built up during annotation and released with the dataset; it is
    also what the verifier matches against, so it is a deliverable, not scratch.
    """
    key = surface.strip().lower()
    return table.get(key, key)


def save_annotations(annotations: Iterable[ReviewAnnotation], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for annotation in annotations:
            fh.write(json.dumps(annotation.to_dict(), ensure_ascii=False) + "\n")


def load_annotations(path: str | Path) -> list[ReviewAnnotation]:
    with Path(path).open(encoding="utf-8") as fh:
        return [ReviewAnnotation.from_dict(json.loads(line)) for line in fh if line.strip()]


def krippendorff_alpha(ratings: Mapping[str, Sequence[object]]) -> float:
    """Krippendorff's alpha for nominal data.

    `ratings` maps a unit id to one value per coder; `None` marks a missing
    rating. Units with fewer than two ratings carry no pairable information and
    are skipped, exactly as the coefficient's definition requires.

    Returns NaN when every rating shares a single value: there is no expected
    disagreement to normalise against, so alpha is undefined rather than 1.0.
    That distinction matters here -- a pilot where both annotators mark almost
    nothing would otherwise look like perfect agreement.
    """
    units = [[v for v in vals if v is not None] for vals in ratings.values()]
    units = [u for u in units if len(u) >= 2]
    if not units:
        raise ValueError("no pairable units: every unit needs at least two ratings")

    n_total = sum(len(u) for u in units)

    observed = 0.0
    for unit in units:
        counts = Counter(unit)
        m = len(unit)
        # Number of disagreeing ordered pairs within the unit, normalised by m-1.
        disagreeing = m * m - sum(c * c for c in counts.values())
        observed += disagreeing / (m - 1)
    d_observed = observed / n_total

    totals = Counter(v for unit in units for v in unit)
    d_expected = (
        n_total * n_total - sum(c * c for c in totals.values())
    ) / (n_total * (n_total - 1))

    if d_expected == 0:
        return math.nan
    return 1.0 - d_observed / d_expected


@dataclass
class AgreementReport:
    alpha: float
    n_reviews: int
    n_events: int
    n_disagreements: int
    disagreements: list[tuple[str, str]]
    verdict: str
    perfect_agreement: bool = False
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)

    def summary(self) -> str:
        alpha_line = (
            "undefined (no rating variance)" if math.isnan(self.alpha)
            else f"{self.alpha:.4f}"
        )
        lines = [
            f"Krippendorff alpha : {alpha_line}",
            f"reviews compared   : {self.n_reviews}",
            f"event decisions    : {self.n_events}",
            f"disagreements      : {self.n_disagreements}",
            f"verdict            : {self.verdict.upper()}",
        ]
        if self.perfect_agreement:
            lines.append(
                "\nNote: annotators agreed on every decision, so there is no "
                "expected disagreement to normalise against and alpha is "
                "undefined. On a real pilot this means the sample was too easy "
                "or too small -- widen it before trusting the gate."
            )
        if self.only_in_a or self.only_in_b:
            lines.append(
                "\nWarning: reviews annotated by only one annotator are left out "
                "of the comparison. Finish them, or confirm they were meant to "
                "be excluded:"
            )
            for side, review_ids in (("A", self.only_in_a), ("B", self.only_in_b)):
                if review_ids:
                    lines.append(f"  only in {side}: {', '.join(review_ids)}")
        return "\n".join(lines)


def agreement_report(
    annotator_a: Mapping[str, set[str]],
    annotator_b: Mapping[str, set[str]],
) -> AgreementReport:
    """Compare two annotators' event sets at the normalised-label level.

    Each (review, event) pair seen by either annotator becomes one binary
    decision -- did this annotator mark this event present? That is the unit
    alpha is computed over, and it is the level the coverage loss operates at.

    Reviews present in only one file cannot be compared; they are listed on the
    report rather than dropped without a trace.
    """
    shared = sorted(set(annotator_a) & set(annotator_b))

    ratings: dict[str, list[object]] = {}
    disagreements: list[tuple[str, str]] = []
    for review_id in shared:
        events_a, events_b = annotator_a[review_id], annotator_b[review_id]
        for event in sorted(events_a | events_b):
            in_a, in_b = int(event in events_a), int(event in events_b)
            ratings[f"{review_id}::{event}"] = [in_a, in_b]
            if in_a != in_b:
                disagreements.append((review_id, event))

    alpha = krippendorff_alpha(ratings) if ratings else math.nan

    # Alpha is undefined when no rating varies, which happens either because
    # there is nothing to compare or because the annotators agreed on
    # everything. Those are opposite situations and must not share a verdict.
    perfect = bool(ratings) and len(disagreements) == 0 and math.isnan(alpha)

    if perfect:
        verdict = "freeze"
    elif math.isnan(alpha):
        verdict = "undefined"
    elif alpha >= FREEZE_THRESHOLD:
        verdict = "freeze"
    elif alpha >= STOP_THRESHOLD:
        verdict = "revise"
    else:
        verdict = "stop"

    return AgreementReport(
        alpha=alpha,
        n_reviews=len(shared),
        n_events=len(ratings),
        n_disagreements=len(disagreements),
        disagreements=disagreements,
        verdict=verdict,
        perfect_agreement=perfect,
        only_in_a=sorted(set(annotator_a) - set(annotator_b)),
        only_in_b=sorted(set(annotator_b) - set(annotator_a)),
    )


def canonicalise(
    events_by_review: Mapping[str, set[str]],
    table: Mapping[str, str],
) -> dict[str, set[str]]:
    """Map every label through the synonym table, returning new sets.

    Agreement is compared on exact strings, so "drowsy" against "drowsiness" is
    otherwise a disagreement. Merging labels can remove disagreements but never
    add one, so report agreement before and after the table -- the gap is how
    much the table did, and a reader can judge whether it did too much.
    """
    return {
        review_id: {normalise_event(label, table) for label in labels}
        for review_id, labels in events_by_review.items()
    }


def load_synonym_table(path: str | Path) -> dict[str, str]:
    """Read a `surface,canonical` CSV into the table `normalise_event` expects.

    A surface form mapped to two different labels is an error rather than
    last-row-wins, which would make the merge depend on row order.
    """
    table: dict[str, str] = {}
    # utf-8-sig: Excel prepends a byte-order mark when it saves CSV.
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not {"surface", "canonical"} <= set(reader.fieldnames or []):
            raise ValueError(
                f"{path}: expected a 'surface,canonical' header, got {reader.fieldnames}"
            )
        for row in reader:
            surface = (row["surface"] or "").strip().lower()
            canonical = (row["canonical"] or "").strip().lower()
            if not surface or not canonical:
                continue
            if table.get(surface, canonical) != canonical:
                raise ValueError(
                    f"{path}: {surface!r} maps to both {table[surface]!r} and {canonical!r}"
                )
            table[surface] = canonical
    return table


# Event slots per review row in the annotation sheet.
N_EVENT_SLOTS = 8

_DONE_VALUES = {"yes", "y", "x", "true", "1", "1.0"}
_SLOT_FIELDS = ("span", "event", "certainty", "attribution")


class AnnotationSheetError(ValueError):
    """Every problem found in a filled sheet, so it can be fixed in one pass."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__(
            f"{len(self.problems)} problem(s) in the annotation sheet:\n  "
            + "\n  ".join(self.problems)
        )


def parse_sheet_rows(
    rows: Iterable[Mapping[str, object]],
    n_slots: int = N_EVENT_SLOTS,
) -> list[ReviewAnnotation]:
    """Turn filled sheet rows into annotations, refusing anything ambiguous.

    Each row is one review, with event slots `span_i`, `event_i`, `certainty_i`
    and `attribution_i` for i in 1..n_slots. Refusals, and why:

    - A review not marked `done` is rejected, never read as "no adverse
      events". `agreement_report` compares only reviews both annotators have,
      so a skipped review would silently leave the comparison.
    - Blank certainty or attribution is rejected, never defaulted. Negation and
      condition-attribution are the failure modes the pilot watches; a default
      would record a decision nobody made.
    - A partly filled slot is rejected, and so is a duplicated review.

    Problems are collected across the whole sheet and raised together.
    """
    certainties = {c.value for c in Certainty}
    attributions = {a.value for a in Attribution}
    annotations: list[ReviewAnnotation] = []
    problems: list[str] = []
    seen: set[str] = set()

    # Row 1 of the sheet is the header, so data starts at row 2.
    for sheet_row, row in enumerate(rows, start=2):
        review_id = _cell(row.get("review_id"))
        if not review_id:
            problems.append(f"row {sheet_row}: missing review_id")
            continue
        where = f"{review_id} (row {sheet_row})"
        if review_id in seen:
            problems.append(f"{where}: duplicate review_id")
            continue
        seen.add(review_id)

        row_problems: list[str] = []
        if _cell(row.get("done")).lower() not in _DONE_VALUES:
            row_problems.append(f"{where}: not marked done")

        events: list[AdverseEvent] = []
        for i in range(1, n_slots + 1):
            span, event, certainty, attribution = (
                _cell(row.get(f"{name}_{i}")) for name in _SLOT_FIELDS
            )
            certainty, attribution = certainty.lower(), attribution.lower()
            if not (span or event or certainty or attribution):
                continue

            issues = [
                f"missing {name}"
                for name, value in zip(_SLOT_FIELDS, (span, event, certainty, attribution))
                if not value
            ]
            if certainty and certainty not in certainties:
                issues.append(f"certainty {certainty!r} is not one of {sorted(certainties)}")
            if attribution and attribution not in attributions:
                issues.append(f"attribution {attribution!r} is not one of {sorted(attributions)}")
            if issues:
                row_problems.append(f"{where} slot {i}: {'; '.join(issues)}")
                continue

            events.append(
                AdverseEvent(
                    span=span,
                    event=normalise_event(event, {}),
                    certainty=Certainty(certainty),
                    attribution=Attribution(attribution),
                )
            )

        problems.extend(row_problems)
        if not row_problems:
            annotations.append(ReviewAnnotation(review_id=review_id, events=events))

    if problems:
        raise AnnotationSheetError(problems)
    return annotations


def _cell(value: object) -> str:
    """A sheet cell as stripped text; an empty cell (None) becomes ''."""
    return "" if value is None else str(value).strip()
