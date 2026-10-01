"""Remove reviewer attribution from summary statements before NLI (spec §5).

Reviews are first-person ("the smell is awful"); summaries attribute
("Some reviewers found the scent overpowering"). NLI with a review as premise
scores the attributed form near zero even when a review says exactly that
(30 Sep diagnostic: 5 of 8 unsupported statements became supported without
the frame). This module rewrites a known frame into the reviews' own register:

- a communication verb with "that" keeps only the reported clause
  ("Some reviewers mention that it broke" -> "It broke");
- otherwise the reviewers become "I" ("Others loved the fragrance" ->
  "I loved the fragrance"), with verb agreement and their -> my;
- "according to reviewers, X" -> "X"; "is considered / regarded as" -> "is".

Only frames built from the word lists below are touched; anything else is
returned unchanged, so the rule can miss frames but should not invent them.
The score using it is a secondary, attribution-robust faithfulness metric;
the registered metric scores statements as written.
"""

from __future__ import annotations

import re

_LEAD = re.compile(
    r"^(?:(?:overall|however|additionally|in addition|moreover|furthermore|also|on the other hand|"
    r"in contrast|notably|meanwhile|similarly|conversely|that said|despite this|interestingly|"
    r"unfortunately|finally|lastly|specifically|in particular|for example|for instance|while|"
    r"whereas|but|and|yet|although|though)\b,?\s+)+",
    re.IGNORECASE,
)
_QUANT = (r"(?:a\s+(?:few|number\s+of|couple\s+of|handful\s+of|majority\s+of)|the\s+majority\s+of|"
          r"some|many|several|most|few|numerous|multiple|all|both|other|one|another|an?|the|these|"
          r"those|certain|various|two|three|four|five|six|seven)")
_NOUN = (r"(?:reviewers?|users?|customers?|people|patients?|buyers?|consumers?|parents?|"
         r"individuals?|purchasers?|clients?|others|reviews)")
_SUBJECT = re.compile(
    rf"^(?:(?P<quant>{_QUANT})\s+(?:of\s+(?:the|these|those)\s+)?)?(?P<noun>{_NOUN})\s+(?P<rest>.+)$"
    rf"|^(?P<bare>some|many|several|others|a few|most)\s+(?P<bare_rest>.+)$",
    re.IGNORECASE,
)
_ACCORDING = re.compile(rf"^according\s+to\s+(?:{_QUANT}\s+)?(?:of\s+the\s+)?{_NOUN},?\s+(?P<rest>.+)$",
                        re.IGNORECASE)
_PASSIVE = re.compile(
    r"\b(is|are|was|were)\s+(?:(?:widely|generally|often|highly|largely|commonly|frequently|"
    r"universally|mostly)\s+)?(?:considered|regarded\s+as|seen\s+as|described\s+as|viewed\s+as|"
    r"reported\s+to\s+be|said\s+to\s+be|thought\s+to\s+be|believed\s+to\s+be|deemed|praised\s+as|"
    r"touted\s+as|perceived\s+as)\s+",
    re.IGNORECASE,
)

# Verbs that report speech or judgement and can take a "that" clause.
_COMM = ["mention", "report", "note", "state", "say", "claim", "feel", "agree", "notice", "observe",
         "indicate", "describe", "suggest", "believe", "think", "warn", "complain", "explain", "add",
         "find", "express", "share", "stress", "emphasize", "emphasise", "highlight", "comment",
         "confirm", "acknowledge", "admit", "caution", "write", "insist", "argue", "point"]
# Other verbs a reviewer is the subject of.
_OTHER = ["experience", "love", "like", "dislike", "hate", "enjoy", "appreciate", "praise",
          "recommend", "prefer", "want", "wish", "need", "use", "try", "take", "see", "lose", "gain",
          "suffer", "struggle", "criticize", "criticise", "question", "rave", "rate", "call",
          "consider", "give", "receive", "buy", "purchase", "return", "stop", "switch", "continue",
          "start", "seem", "value", "develop", "get", "avoid", "tolerate", "achieve", "benefit",
          "swear", "sleep", "remain", "become", "prefer"]
_IRREGULAR = {"say": {"said"}, "feel": {"felt"}, "think": {"thought"}, "find": {"found"},
              "write": {"wrote", "written"}, "take": {"took", "taken"}, "see": {"saw", "seen"},
              "lose": {"lost"}, "get": {"got", "gotten"}, "give": {"gave", "given"},
              "buy": {"bought"}, "swear": {"swore", "sworn"}, "sleep": {"slept"},
              "stop": {"stopped"}, "admit": {"admitted"}, "prefer": {"preferred"},
              "become": {"became"}, "rate": {"rated"}}
_AUX_AGREEMENT = {"has": "have", "have": "have", "had": "had", "is": "am", "are": "am", "was": "was",
                  "were": "was", "does": "do", "do": "do", "did": "did", "can": "can", "could": "could",
                  "will": "will", "would": "would", "may": "may", "might": "might",
                  "should": "should", "must": "must"}
_ADVERBS = {"also", "even", "still", "all", "both", "never", "always", "just", "further", "not",
            "often", "initially", "eventually", "previously", "originally"}


def _third_person(verb: str) -> str:
    if re.search(r"(s|sh|ch|x|z|o)$", verb):
        return verb + "es"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    return verb + "s"


def _past(verb: str) -> str:
    if verb.endswith("e"):
        return verb + "d"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ied"
    return verb + "ed"


def _forms(verbs: list[str]) -> tuple[set[str], dict[str, str]]:
    """All forms of each verb, and a map from third-person forms back to the base."""
    forms, third = set(), {}
    for v in verbs:
        forms |= {v, _third_person(v), _past(v)} | _IRREGULAR.get(v, set())
        third[_third_person(v)] = v
    return forms, third


_COMM_FORMS, _COMM_THIRD = _forms(_COMM)
_OTHER_FORMS, _OTHER_THIRD = _forms(_OTHER)
_THIRD = {**_OTHER_THIRD, **_COMM_THIRD}

_PRONOUNS = [(r"\bthey are\b", "I am"), (r"\bthey were\b", "I was"), (r"\bthey're\b", "I'm"),
             (r"\bthey've\b", "I've"), (r"\bthey'd\b", "I'd"), (r"\bthey'll\b", "I'll"),
             (r"\bthemselves\b", "myself"), (r"\btheirs\b", "mine"), (r"\btheir\b", "my"),
             (r"\bthem\b", "me"), (r"\bthey\b", "I")]


def _first_person(text: str) -> str:
    for pattern, repl in _PRONOUNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def _word(token: str) -> str:
    return re.sub(r"[^\w']", "", token).lower()


def _is_adverb(word: str) -> bool:
    return word in _ADVERBS or (len(word) > 4 and word.endswith("ly"))


def _rewrite_frame(rest: str, reviews_subject: bool) -> str | None:
    """`rest` follows a reviewer subject. Return the rewrite, or None if no frame is recognised."""
    words = rest.split()
    i = 0
    while i < len(words) and _is_adverb(_word(words[i])):
        i += 1
    if i >= len(words):
        return None
    first = _word(words[i])
    if first not in _AUX_AGREEMENT and first not in _COMM_FORMS and first not in _OTHER_FORMS:
        return None

    # A communication verb (possibly after an auxiliary and adverbs) followed by "that".
    j = i + 1 if first in _AUX_AGREEMENT else i
    while j < len(words) and _is_adverb(_word(words[j])):
        j += 1
    if j < len(words) and _word(words[j]) in _COMM_FORMS:
        k = j + 1
        if k < len(words) and _word(words[k]) == "out":  # "point out that"
            k += 1
        if k < len(words) and _word(words[k]) == "that":
            clause = " ".join(words[k + 1:])
            return _first_person(clause) if len(clause.split()) >= 2 else None

    if reviews_subject:  # "Most reviews are positive" is about the balance of opinion
        return None
    if len(words) - i - 1 < 1:  # "Some reviewers agreed": nothing to check against the reviews
        return None
    verb = words[i]
    fixed = _AUX_AGREEMENT.get(first) or _THIRD.get(first) or verb
    words = words[:i] + [fixed if first in _AUX_AGREEMENT or first in _THIRD else verb] + words[i + 1:]
    return "I " + _first_person(" ".join(words))


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def deattribute(text: str) -> str:
    """`text` with a reviewer-attribution frame removed; unchanged if no known frame is found."""
    body = _LEAD.sub("", text.strip())
    out = None

    m = _ACCORDING.match(body)
    if m:
        out = m.group("rest")
    else:
        m = _SUBJECT.match(body)
        # A bare singular noun is an adjective ("individual experiences vary"), not a reviewer.
        bare_singular = (m and m.group("noun") and not m.group("quant")
                         and not re.search(r"(s|people)$", m.group("noun"), re.IGNORECASE))
        if m and not bare_singular:
            if m.group("noun"):
                out = _rewrite_frame(m.group("rest"), m.group("noun").lower() == "reviews")
            else:
                out = _rewrite_frame(m.group("bare_rest"), False)

    candidate = out if out is not None else body
    passive = _PASSIVE.sub(lambda p: p.group(1) + " ", candidate)
    if out is None and passive == candidate:
        return text
    return _capitalise(passive.strip())
