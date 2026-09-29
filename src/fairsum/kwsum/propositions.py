"""Summary text to scoreable propositions (spec §5).

REFER split summaries into single-opinion propositions with GPT-4o-mini. This
rule-based splitter is the local stand-in: sentences, then clauses at contrast
markers. Three kinds of text are set aside and never scored as opinions:

* **Pure count statements** ("Out of 8 reviews, 6 are positive and 2 are
  negative"). They are checked separately for accuracy.
* **Meta text**: reasoning steps, per-review notes, instruction echoes,
  headings, and everything before a final "Summary:" header.
* **Fragments** under three words.

A clause that carries a count *and* an opinion ("Two reviewers reported
nausea") is kept, with the number replaced by "some". REFER-style prompts
attribute opinions with numbers far more often than other prompts do (final
review, finding C1), so dropping these clauses would score conditions on
different subsets of their own summaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_F = re.IGNORECASE
_S = r"[ \t]+"  # counts never cross a line break
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUM = r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
_NUMG = r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
_NOUN = r"(?:reviews?|reviewers?|users?|customers?|patients?|people|buyers?)"
_NOUNG = r"(reviews?|reviewers?|users?|customers?|patients?|people|buyers?)"
_POL = r"(?:positive|negative|neutral|mixed)"

# Count phrases: each can appear in a pure count statement or inside an opinion.
_PAREN_TOTAL = re.compile(
    rf"[ \t]*\([ \t]*{_NUM}{_S}(?:out{_S})?of{_S}(?:the{_S}|all{_S})?{_NUM}(?:{_S}{_NOUN})?[ \t]*\)", _F)
_TOTAL = re.compile(
    rf"\b{_NUMG}{_S}(?:out{_S})?of{_S}(?:the{_S}|all{_S})?{_NUM}\b(?:{_S}{_NOUNG}\b)?", _F)
_OUT_OF = re.compile(rf"\b(?:out{_S}of|of{_S}the|of{_S}all){_S}{_NUM}\b(?:{_S}{_NOUN}\b)?[ \t]*,?", _F)
_NUM_POL_NOUN = re.compile(rf"\b{_NUMG}{_S}({_POL}){_S}{_NOUNG}\b", _F)
_NUM_NOUN = re.compile(rf"\b{_NUMG}{_S}{_NOUNG}\b", _F)
_NUM_VERB_POL = re.compile(
    rf"\b{_NUMG}{_S}(?:(?:are|were|is|was){_S})?(?:[a-z]+ly{_S})?({_POL})\b", _F)
_COUNT_PHRASES = (_PAREN_TOTAL, _TOTAL, _OUT_OF, _NUM_POL_NOUN, _NUM_NOUN, _NUM_VERB_POL)

_PARSE = re.compile(
    rf"\b{_NUMG}\b(?:{_S}(?:out{_S})?of{_S}(?:the{_S}|all{_S})?{_NUM}\b)?(?:{_S}[a-z']+){{0,4}}?{_S}(positive|negative)\b",
    _F)
_LABEL = re.compile(
    rf"^\W*(positive|negative)(?:{_S}(?:reviews?|opinions?|feedback|experiences?))?[ \t]*[:=\-][ \t]*{_NUMG}(?:[ \t]*/[ \t]*{_NUM})?\b",
    _F)
_LABEL_PREFIX = re.compile(
    rf"^{_POL}(?:{_S}(?:reviews?|opinions?|feedback|experiences?|aspects?))?[ \t]*[:\-–][ \t]*", _F)

_SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")
_CONTRAST = re.compile(
    r"\s*;\s*|,?\s+(?:but|however|although|though|whereas|while)\s+", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)](?=\s|$))\s*")
_MARKDOWN = re.compile(r"[*_#`]+")
_DROP_META = re.compile(
    r"^(?:step\s*\d+|let'?s\b|let us\b|let me\b|to (?:determine|summari[sz]e|answer)\b"
    r"|first,? (?:let|we|i)\b|review\s*#?\s*\d+\b|\d+[.)]?\s*(?:positive|negative|neutral)\b)",
    re.IGNORECASE)
_ECHO = re.compile(
    r"how many|are positive,\s*negative|generate a balanced summary|the summary of the reviews is",
    re.IGNORECASE)
_PREFIX_META = re.compile(
    r"^(?:here(?:'s| is)\b[^:]*|below is\b[^:]*|(?:the |a |overall |final )?summary\b[^:]*):\s*",
    re.IGNORECASE)
_SUMMARY_HEADER = re.compile(
    r"^[ \t]*[#*_]*[ \t]*(?:final[ \t]+|overall[ \t]+)?summary\b[^\n:.]*?[*_ \t]*(?::[*_ \t]*|$)",
    re.IGNORECASE | re.MULTILINE)
_TERMINATED = re.compile(r"[.!?][\"')\]*]*\s*$")
_HEADING_ONLY = re.compile(
    r"^(?:summary|overview|conclusion|assessment|verdict|pros|cons|key points|highlights)$",
    re.IGNORECASE)

MIN_PROP_WORDS = 3
MIN_OPINION_WORDS = 3  # content words a counted clause needs to count as an opinion
MIN_AFTER_HEADER_WORDS = 5
_COUNT_FILLER = frozenset({
    "review", "reviews", "reviewer", "reviewers", "user", "users", "customer", "customers",
    "patient", "patients", "people", "buyer", "buyers", "total", "overall", "rest",
    "remaining", "others", "other", "majority", "minority", "most", "some", "few",
})
_STOP = frozenset(ENGLISH_STOP_WORDS)


@dataclass(frozen=True)
class SplitSummary:
    propositions: tuple[str, ...]
    counts: tuple[tuple[int, str], ...]
    n_count_statements: int
    n_meta: int


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else _WORD_NUMBERS[token.lower()]


def parse_counts(text: str) -> list[tuple[int, str]]:
    """(count, polarity) pairs, e.g. "6 of the 8 reviews are positive" -> [(6, 'positive')].

    `split_summary` applies this only to pure count statements and never to
    meta text, so numbers in product titles or echoed instructions are not
    read as counts.
    """
    text = text.replace("’", "'")
    label = _LABEL.match(text)
    if label:
        return [(_to_int(label.group(2)), label.group(1).lower())]
    return [(_to_int(m.group(1)), m.group(2).lower()) for m in _PARSE.finditer(text)]


def _has_count_phrase(piece: str) -> bool:
    return any(rx.search(piece) for rx in _COUNT_PHRASES)


def _opinion_words(piece: str) -> int:
    """Content words left once every count phrase is removed."""
    for rx in _COUNT_PHRASES:
        piece = rx.sub(" ", piece)
    tokens = re.findall(r"[a-z']+", piece.lower())
    return sum(1 for t in tokens if t not in _STOP and t not in _COUNT_FILLER)


def _plural(noun: str) -> str:
    noun = noun.lower()
    return noun if noun.endswith("s") or noun == "people" else noun + "s"


def _generic_subject(piece: str) -> str:
    """Replace numbered subjects by "some": "Two reviewers said X" -> "Some reviewers said X"."""
    s = _PAREN_TOTAL.sub("", piece)
    s = _TOTAL.sub(lambda m: f"some {_plural(m.group(2))}" if m.group(2) else "some", s)
    s = _OUT_OF.sub("", s)
    s = _NUM_POL_NOUN.sub(lambda m: f"some {m.group(2).lower()} {_plural(m.group(3))}", s)
    s = _NUM_NOUN.sub(lambda m: f"some {_plural(m.group(2))}", s)
    s = _NUM_VERB_POL.sub(lambda m: "some" + m.group(0)[len(m.group(1)):], s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip(" ,;:-")
    return s[:1].upper() + s[1:]


def _after_last_summary_header(text: str) -> tuple[str, bool]:
    """Keep only what follows the last "Summary:" header, if enough follows it."""
    headers = list(_SUMMARY_HEADER.finditer(text))
    if not headers:
        return text, False
    tail = text[headers[-1].end():].strip()
    if len(tail.split()) < MIN_AFTER_HEADER_WORDS:
        return text, False
    return tail, True


def _is_heading(raw_line: str, cleaned: str) -> bool:
    """A short unpunctuated line in title case, or a bare section word."""
    if _TERMINATED.search(raw_line) or re.search(r"\d", cleaned):
        return False
    if _HEADING_ONLY.match(cleaned.strip()):
        return True
    words = cleaned.split()
    content = [w for w in words if w.lower() not in _STOP]
    if not 2 <= len(content) <= 6 or len(words) > 8:
        return False
    return sum(w[:1].isupper() for w in content) / len(content) >= 0.6


def split_summary(text: str, drop_unterminated_tail: bool = False) -> SplitSummary:
    """Propositions, parsed counts, and how much count and meta text was set aside.

    `drop_unterminated_tail` drops a final sentence cut off mid-way, which is
    what a generation that hit its token cap ends with.
    """
    text = (text or "").replace("’", "'").strip()
    text, trimmed = _after_last_summary_header(text)
    n_meta = int(trimmed)
    sentences = [s for s in _SENT_END.split(text) if s.strip()]
    if drop_unterminated_tail and sentences and not _TERMINATED.search(sentences[-1]):
        sentences = sentences[:-1]

    props: list[str] = []
    counts: list[tuple[int, str]] = []
    n_count = 0
    for raw in sentences:
        sentence = _MARKDOWN.sub("", _BULLET.sub("", raw)).strip()
        if not sentence:
            continue
        if _LABEL.match(sentence):
            n_count += 1
            counts.extend(parse_counts(sentence))
            continue
        if (_DROP_META.match(sentence) or _DROP_META.match(_MARKDOWN.sub("", raw).strip())
                or _ECHO.search(sentence) or sentence.endswith(":") or _is_heading(raw, sentence)):
            n_meta += 1
            continue
        prefix = _PREFIX_META.match(sentence)
        if prefix:
            n_meta += 1
            sentence = sentence[prefix.end():].strip()
        sentence = _LABEL_PREFIX.sub("", sentence)
        for piece in _CONTRAST.split(sentence):
            piece = piece.strip(" ,.;:-")
            if not piece:
                continue
            if _has_count_phrase(piece):
                if _opinion_words(piece) < MIN_OPINION_WORDS:
                    n_count += 1
                    counts.extend(parse_counts(piece))
                    continue
                piece = _generic_subject(piece)
            if len(piece.split()) >= MIN_PROP_WORDS:
                props.append(piece)
    return SplitSummary(tuple(props), tuple(counts), n_count, n_meta)
