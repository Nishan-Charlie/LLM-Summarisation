"""Does a summary convey a given adverse event?

Two entry points that must never be confused:

  `measure_coverage` / `coverage_loss` -- the evaluation path. Takes gold
  annotated events and produces the loss the conformal layer calibrates against.

  `score_coverage` -- the deployment path. Label-free, drives best-of-k
  selection and the acceptance threshold at inference time.

The entailment model is injected. It MUST be a different model family from
whatever generated the summary: v1's central defect was a 1B model grading its
own output (vault note `F1 Degenerate Judge Rows`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, Sequence

# Above this entailment probability an event counts as conveyed; below the
# contradiction bound it counts as absent. The band between them is ABSTAIN.
ENTAILMENT_THRESHOLD = 0.5
ABSTAIN_BAND = 0.15


class Verdict(Enum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    ABSTAIN = "abstain"


class EntailmentModel(Protocol):
    """Returns a verdict plus whatever raw output produced it."""

    def __call__(self, summary: str, event: str) -> tuple[Verdict, str]: ...


@dataclass
class CoverageReport:
    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    abstained: list[str] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.covered) + len(self.missing) + len(self.abstained)


def measure_coverage(
    summary: str,
    events: Sequence[str],
    model: EntailmentModel,
) -> CoverageReport:
    """Check each gold adverse event against the summary, retaining raw output."""
    report = CoverageReport()
    for event in events:
        verdict, raw = model(summary, event)
        report.raw[event] = raw
        if verdict is Verdict.COVERED:
            report.covered.append(event)
        elif verdict is Verdict.ABSTAIN:
            report.abstained.append(event)
        else:
            report.missing.append(event)
    return report


def coverage_loss(
    report: CoverageReport,
    abstain_counts_as_missing: bool = True,
) -> float:
    """Fraction of reported adverse events the summary failed to convey.

    Bounded in [0, 1] and monotone in the omissions, which is what
    `fairsum.conformal` requires. An instance with no adverse events has nothing
    to drop and scores 0 rather than raising.

    Abstentions count against coverage by default. That is the conservative
    reading -- an event the verifier could not decide on may well be missing --
    and it means verifier noise inflates the measured risk rather than hiding it.
    """
    if report.n_events == 0:
        return 0.0
    failed = len(report.missing)
    if abstain_counts_as_missing:
        failed += len(report.abstained)
    return failed / report.n_events


def score_coverage(
    summary: str,
    events: Sequence[str],
    model: EntailmentModel,
) -> float:
    """Label-free coverage score in [0, 1], higher is better.

    In deployment `events` are the verifier's own extracted candidates rather
    than gold annotations; the arithmetic is identical, the provenance is not.
    """
    if not events:
        return 1.0
    return 1.0 - coverage_loss(measure_coverage(summary, events, model))


def select_best(
    candidates: Sequence[str],
    scores: Sequence[float],
) -> tuple[str, float, int]:
    """Best-of-k selection. Ties go to the first candidate, deterministically."""
    if len(candidates) == 0:
        raise ValueError("select_best needs at least one candidate")
    if len(candidates) != len(scores):
        raise ValueError(
            f"candidates and scores must have the same length "
            f"(got {len(candidates)} and {len(scores)})"
        )
    best_idx = max(range(len(scores)), key=lambda i: (scores[i], -i))
    return candidates[best_idx], scores[best_idx], best_idx


def build_nli_verifier(
    model_name: str = "microsoft/deberta-large-mnli",
    device: str | None = None,
    entail_threshold: float = ENTAILMENT_THRESHOLD,
    abstain_band: float = ABSTAIN_BAND,
) -> Callable[[str, str], tuple[Verdict, str]]:
    """An `EntailmentModel` backed by a natural-language-inference classifier.

    Premise is the summary, hypothesis is "The patient reported {event}." Small
    enough (~400M) to sit alongside a 1B-3B generator in 8 GB, and from a
    different family than the Llama/Qwen generators by construction.

    Imports torch lazily so the pure-logic modules stay importable without a GPU
    stack present.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    label_names = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    try:
        entail_idx = next(i for i, n in enumerate(label_names) if "entail" in n)
    except StopIteration as exc:
        raise ValueError(
            f"{model_name} has labels {label_names} with no entailment class; "
            "it is not an NLI model"
        ) from exc

    def verify(summary: str, event: str) -> tuple[Verdict, str]:
        hypothesis = f"The patient reported {event}."
        inputs = tokenizer(
            summary, hypothesis, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            probs = model(**inputs).logits.softmax(-1)[0]
        p_entail = float(probs[entail_idx])
        raw = " ".join(f"{n}={float(p):.4f}" for n, p in zip(label_names, probs))

        if p_entail >= entail_threshold + abstain_band:
            verdict = Verdict.COVERED
        elif p_entail <= entail_threshold - abstain_band:
            verdict = Verdict.NOT_COVERED
        else:
            verdict = Verdict.ABSTAIN
        return verdict, raw

    return verify
