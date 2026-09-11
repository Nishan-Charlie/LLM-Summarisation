"""Summarization prompts, and parsing of what comes back.

Every prompt ends by requiring a FINAL SUMMARY: line. Only the text after it is
scored, so a prompt that asks the model to enumerate counts or run a checklist
first is not penalised for spending tokens on that scaffold
(`F4 Design Confounds` #4).

A response with no marker is a parse failure carrying its own status. There is
deliberately no fallback to the raw text: silently accepting whatever came back
is how v1's judge failures became fairness scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from fairsum.instances import Instance
from fairsum.metrics import EFFECTIVENESS_LABELS

FINAL_SUMMARY_MARKER = "FINAL SUMMARY:"
REVIEW_SEPARATOR = " || "

_MARKER_RE = re.compile(r"^\s*final\s+summary\s*:\s*$|final\s+summary\s*:", re.IGNORECASE)
_TRAILING_BOILERPLATE = re.compile(
    r"(\*{0,2}\s*final\s+summary\s*\*{0,2}\s*:?\s*)$", re.IGNORECASE
)


class ParseStatus(Enum):
    OK = "ok"
    NO_MARKER = "no_marker"
    EMPTY = "empty"


@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    summary: str | None = None


def parse_final_summary(raw: str, requires_marker: bool = True) -> ParseResult:
    """Extract the scored summary from a raw generation.

    Splits on the last marker, since a model that drafts and then revises emits
    the marker more than once and the final block is the one it stands behind.

    `requires_marker=False` is for prompts that produce no reasoning scaffold,
    where the whole completion is the summary. That is a per-prompt contract
    declared in `PROMPT_SPECS`, not a fallback: a prompt that promises a marker
    and omits it still fails.
    """
    if not raw:
        return ParseResult(ParseStatus.NO_MARKER if requires_marker else ParseStatus.EMPTY)

    matches = list(_MARKER_RE.finditer(raw))
    if not matches:
        if requires_marker:
            return ParseResult(ParseStatus.NO_MARKER)
        body = raw.strip()
        return ParseResult(ParseStatus.OK, body) if body else ParseResult(ParseStatus.EMPTY)

    tail = raw[matches[-1].end() :]
    tail = _TRAILING_BOILERPLATE.sub("", tail).strip()
    if not tail:
        return ParseResult(ParseStatus.EMPTY)
    return ParseResult(ParseStatus.OK, tail)


def _source_block(instance: Instance) -> str:
    return REVIEW_SEPARATOR.join(instance.texts)


def _header(instance: Instance) -> str:
    """Name the indication only when the instance actually has a single one.

    v1 asserted the drug's first recorded condition for every instance, which
    was false whenever the reviews spanned indications -- and on real Druglib
    that is most of them, at 4.5 indications per drug on average. A prompt that
    makes no indication claim cannot make a false one.
    """
    indication = (
        f", used to treat {instance.condition_name}" if instance.condition_name else ""
    )
    return (
        f"Below are {len(instance.reviews)} patient reviews of the same "
        f"medication{indication}. Each review is separated by ||.\n\n"
        f"{_source_block(instance)}"
    )


def _plain_footer() -> str:
    """Closing for prompts with no reasoning phase: the reply IS the summary."""
    return "Reply with the summary only, as one paragraph of about 60 words."


def _marked_footer() -> str:
    """Closing for prompts with a reasoning phase, which must be stripped.

    The prompt must NOT end on the bare marker: doing so makes the model
    continue straight into the summary without echoing it, and the completion
    then looks like a parse failure. Measured on Llama-3.2-1B, that cost 44-69%
    of otherwise-good generations.
    """
    return (
        f"When you have finished reasoning, write the line {FINAL_SUMMARY_MARKER} "
        f"and then, on the same line, the summary as one paragraph of about 60 "
        f"words. You must write those two words before the summary."
    )


def direct_prompt(instance: Instance) -> str:
    """Baseline. No instruction to count or balance perspectives."""
    return (
        f"{_header(instance)}\n\n"
        f"Write a short summary containing the salient information.\n\n"
        f"{_plain_footer()}"
    )


def refer_prompt(instance: Instance) -> str:
    """Zhang et al.'s REFER: count the perspectives, then summarize in proportion."""
    levels = ", ".join(EFFECTIVENESS_LABELS)
    return (
        f"{_header(instance)}\n\n"
        f"First determine how many of the {len(instance.reviews)} reviews report each of "
        f"these effectiveness levels: {levels}.\n"
        f"Then write a balanced summary reflecting how frequently each level occurs.\n\n"
        f"{_marked_footer()}"
    )


def ssfp_prompt(instance: Instance) -> str:
    """Safety-Salience and Frequency Prompting."""
    levels = ", ".join(EFFECTIVENESS_LABELS)
    return (
        f"{_header(instance)}\n\n"
        f"First determine how many of the {len(instance.reviews)} reviews report each of "
        f"these effectiveness levels: {levels}.\n\n"
        f"Then identify the adverse experiences reported: side effects, poor "
        f"tolerability, lack of effectiveness, worsening symptoms, or stopping "
        f"treatment.\n\n"
        f"Weigh both how often each experience appears and how safety-relevant it is. "
        f"An adverse experience must not be dropped merely because only one patient "
        f"reports it, and must not be exaggerated beyond what the reviews support.\n\n"
        f"Do not add information absent from the reviews, and do not give medical "
        f"advice.\n\n"
        f"{_marked_footer()}"
    )


def ecvp_prompt(instance: Instance) -> str:
    """Evidence-Coverage Verification Prompting: draft, self-check, then emit."""
    levels = ", ".join(EFFECTIVENESS_LABELS)
    return (
        f"{_header(instance)}\n\n"
        f"First determine how many of the {len(instance.reviews)} reviews report each of "
        f"these effectiveness levels: {levels}.\n\n"
        f"Then draft a balanced summary and verify it against these questions:\n"
        f"1. Is the dominant patient experience represented?\n"
        f"2. Is the minority patient experience represented?\n"
        f"3. Is any reported adverse effect missing?\n"
        f"4. Is any experience emphasised beyond its frequency?\n"
        f"5. Is every statement supported by at least one review?\n\n"
        f"Repair whatever the check exposes, then emit only the corrected summary.\n\n"
        f"{_marked_footer()}"
    )


def aspect_aware_prompt(instance: Instance) -> str:
    """Aspect coverage without an explicit counting step."""
    return (
        f"{_header(instance)}\n\n"
        f"Cover the main medical aspects the {len(instance.reviews)} reviews raise: "
        f"effectiveness or symptom relief, treatment benefits, adverse effects, how "
        f"tolerable those effects were, onset or duration, and overall experience.\n\n"
        f"Do not over-focus on the majority experience. Include minority adverse "
        f"effects where they are clinically important. Add nothing the reviews do "
        f"not state.\n\n"
        f"{_plain_footer()}"
    )


def refer_aspect_aware_prompt(instance: Instance) -> str:
    """REFER's counting step combined with aspect coverage."""
    levels = ", ".join(EFFECTIVENESS_LABELS)
    return (
        f"{_header(instance)}\n\n"
        f"First determine how many of the {len(instance.reviews)} reviews report each of "
        f"these effectiveness levels: {levels}.\n\n"
        f"Then identify the main aspects raised: effectiveness or symptom relief, "
        f"treatment benefits, adverse effects, tolerability, onset or duration, and "
        f"overall experience.\n\n"
        f"Write a summary that reflects the frequency of each effectiveness level, "
        f"covers both benefit and adverse aspects, and keeps clinically important "
        f"minority adverse effects.\n\n"
        f"{_marked_footer()}"
    )


def build_repair_prompt(
    instance: Instance,
    draft: str,
    missing_events: Sequence[str],
) -> str:
    """Sequential test-time scaling: name the omissions and ask for a revision.

    The verifier returns a *set* of missing events rather than a scalar score,
    which is the asymmetry H2 predicts will make repair beat selection per FLOP.
    """
    if not missing_events:
        raise ValueError("no missing events to repair; do not call this on a clean draft")

    listed = "\n".join(f"- {event}" for event in missing_events)
    return (
        f"{_header(instance)}\n\n"
        f"Here is a draft summary of those reviews:\n\n{draft}\n\n"
        f"The draft omits these adverse experiences, each reported in at least one "
        f"review:\n{listed}\n\n"
        f"Revise the summary so it conveys them, using wording proportional to how "
        f"often each occurs. Keep what the draft already covers, stay about the same "
        f"length, and add nothing the reviews do not state.\n\n"
        f"{_plain_footer()}"
    )


@dataclass(frozen=True)
class PromptSpec:
    builder: Callable[[Instance], str]
    #: True when the prompt asks the model to reason before summarizing, so the
    #: scaffold must be stripped and a missing marker is a real parse failure.
    requires_marker: bool


PROMPT_SPECS: dict[str, PromptSpec] = {
    "direct": PromptSpec(direct_prompt, requires_marker=False),
    "refer": PromptSpec(refer_prompt, requires_marker=True),
    "ssfp": PromptSpec(ssfp_prompt, requires_marker=True),
    "ecvp": PromptSpec(ecvp_prompt, requires_marker=True),
    "aspect_aware": PromptSpec(aspect_aware_prompt, requires_marker=False),
    "refer_aspect_aware": PromptSpec(refer_aspect_aware_prompt, requires_marker=True),
}

PROMPT_BUILDERS: dict[str, Callable[[Instance], str]] = {
    name: spec.builder for name, spec in PROMPT_SPECS.items()
}


def requires_marker(method: str) -> bool:
    """Whether `method`'s output must carry a FINAL SUMMARY: marker."""
    return PROMPT_SPECS[method].requires_marker
