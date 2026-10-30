"""Part B of the human evaluation: whole summaries judged by people (amendments 9 and 10).

A set is one skewed test collection with two summaries by the same model, `direct` and
`kw_strat`, shown as A and B in random order. From each summary alone, annotators say which split
of opinion it conveys (Q2). With the reviews then shown grouped by rating, they judge how much of
the minority's content each summary conveys (Q1), whether it says anything no review supports
(Q3) and how readable it is (Q4). The details these amendments leave open are fixed in
docs/design/2026-10-30-kw-partb-analysis.md, written before any Part B label existed.

Blinding: the page sees topics, reviews and summaries only. The key that links a set to its
collection, model and prompt order stays in results/kw/<run>/human_partb_key.json.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as st

from fairsum.annotation import krippendorff_alpha, krippendorff_alpha_distance
from fairsum.kwsum.collections import SKEWED_LEVELS, Collection
from fairsum.kwsum.compliance import SCORED_STATUSES
from fairsum.kwsum.scoring import SCORER_VERSION
from fairsum.kwsum.stats import cluster_bootstrap_ci, wilcoxon_p

SEED = 2026
PROMPTS = ("direct", "kw_strat")
SLOTS = ("A", "B")
PRACTICE_COLLECTION = "drugs-test-pos88-000"
GROUPS = (("shared", 10), ("g1", 35), ("g2", 35))
ALPHA_MIN = 0.667

Q1_VALUES = ("none", "some", "most")
Q2_VALUES = ("almost_all_negative", "mostly_negative", "evenly_split", "mostly_positive",
             "almost_all_positive")
Q3_VALUES = ("yes", "no")
Q4_VALUES = ("clear", "minor_problems", "hard_to_read")
QUESTIONS = {"q1": Q1_VALUES, "q2": Q2_VALUES, "q3": Q3_VALUES, "q4": Q4_VALUES}
Q1_CODE = {"none": 0.0, "some": 0.5, "most": 1.0}
# Perceived share of positive opinion: amendment 9's coding, and amendment 10's alternative that
# matches the true splits.
CODINGS = {"a9": dict(zip(Q2_VALUES, (0.05, 0.25, 0.5, 0.75, 0.95))),
           "alt": dict(zip(Q2_VALUES, (0.125, 0.25, 0.5, 0.75, 0.875)))}
RATING_RULE = {"amazon": ("4 or 5 stars", "1 or 2 stars"),
               "drugs": ("rated 7 to 10 out of 10", "rated 1 to 4 out of 10")}
SCORE_FIELDS = ("recall_minority", "recall_minority_deattr", "unsupported_rate",
                "unsupported_rate_deattr", "majority_amplification",
                "majority_amplification_polarity_only")

GUIDE = [
    "HOW TO JUDGE A SET (about 3 minutes each)",
    "",
    "Each set has two short summaries, A and B, of the same 8 reviews of one product or drug.",
    "Step 1. Read summary A on its own and answer: from this summary alone, how do the reviewers",
    "        feel overall? Then do the same for summary B. You do not see the reviews yet.",
    "Step 2. Open the reviews. They are grouped into positive and negative reviews by their rating.",
    "        Your step 1 answers are now fixed. For each summary, answer the three questions below.",
    "",
    "Overall feeling (step 1)",
    "  almost all negative / mostly negative / evenly split / mostly positive / almost all positive",
    "  Judge what the summary conveys, not what you expect the reviews to say.",
    "",
    "Minority coverage (step 2)",
    "  How much of what the smaller group of reviewers says does the summary convey?",
    "  The page names the smaller group: the negative or the positive reviews.",
    "  none: nothing the smaller group says appears in the summary.",
    "  some: part of it appears.",
    "  most: most or all of it appears.",
    "",
    "Unsupported content (step 2)",
    "  Does the summary say anything no review supports, including a wrong number? yes or no.",
    "  A note is optional, e.g. the statement that has no support.",
    "",
    "Readability (step 2): clear / minor problems / hard to read.",
    "",
    "You do not see which model or prompt wrote a summary. Please don't compare answers with",
    "other annotators until everyone has finished.",
]


@dataclass(frozen=True)
class PairSet:
    set_id: str
    group: str
    collection_id: str
    model: str
    dataset: str
    level: str
    entity: str
    a_condition: str
    summary_a: str
    summary_b: str

    def condition(self, slot: str) -> str:
        return self.a_condition if slot == "A" else other_prompt(self.a_condition)

    def slot(self, condition: str) -> str:
        return "A" if condition == self.a_condition else "B"


def other_prompt(condition: str) -> str:
    return PROMPTS[1] if condition == PROMPTS[0] else PROMPTS[0]


def majority_side(level: str) -> str:
    return "positive" if level.startswith("pos") else "negative"


def true_strength(level: str) -> int:
    """How one-sided the true split is: 2 for 7:1 and 1:7 ("almost all"), 1 for 6:2 and 2:6."""
    return 2 if level.endswith("88") else 1


def true_majority_share(level: str) -> float:
    return 0.875 if level.endswith("88") else 0.75


# --- the sample -------------------------------------------------------------------------------

def eligible_pairs(score_rows: Iterable[Mapping], scorer_version: str = SCORER_VERSION) -> set:
    """(collection, model) pairs whose `direct` and `kw_strat` summaries are both scored outputs."""
    status = {}
    for r in score_rows:
        if (r.get("scorer_version") != scorer_version or r.get("split", "test") != "test"
                or r["condition"] not in PROMPTS or r["level"] not in SKEWED_LEVELS):
            continue
        status[(r["collection_id"], r["model"], r["condition"])] = r["status"]
    return {(cid, model) for (cid, model, cond), s in status.items()
            if cond == PROMPTS[0] and s in SCORED_STATUSES
            and status.get((cid, model, PROMPTS[1])) in SCORED_STATUSES}


def sample_sets(score_rows: Iterable[Mapping], generations: Iterable[Mapping],
                collections: Mapping[str, Collection], exclude: Iterable[str] = (),
                models: Sequence[str] | None = None, seed: int = SEED, per_model: int = 2,
                scorer_version: str = SCORER_VERSION,
                groups: Sequence[tuple[str, int]] = GROUPS) -> list[PairSet]:
    """The registered sample: per (dataset, skewed level), `per_model` collections per model, no
    collection twice, none from `exclude` or the practice set. Set ids and groups are assigned
    after a shuffle, so neither says anything about model, dataset or level."""
    ok = eligible_pairs(score_rows, scorer_version)
    texts = {(g["collection_id"], g["model"], g["condition"]): g["raw"] for g in generations
             if g["condition"] in PROMPTS and g.get("split", "test") == "test"}
    excluded = set(exclude) | {PRACTICE_COLLECTION}
    models = sorted(models or {m for _, m in ok})
    rng = random.Random(seed)
    cells = sorted({(c.dataset, c.level) for c in collections.values()
                    if c.split == "test" and c.level in SKEWED_LEVELS})
    chosen = []
    for dataset, level in cells:
        in_cell = sorted(cid for cid, c in collections.items()
                         if (c.dataset, c.level, c.split) == (dataset, level, "test") and cid not in excluded)
        used: set[str] = set()
        order = list(models)
        rng.shuffle(order)
        for model in order:
            free = [cid for cid in in_cell if cid not in used and (cid, model) in ok
                    and all((cid, model, p) in texts for p in PROMPTS)]
            if len(free) < per_model:
                raise ValueError(f"{dataset} {level} {model}: {len(free)} eligible collections, "
                                 f"{per_model} needed")
            for cid in rng.sample(free, per_model):
                used.add(cid)
                chosen.append((cid, model, rng.choice(PROMPTS)))
    if len(chosen) != sum(n for _, n in groups):
        raise ValueError(f"{len(chosen)} sets drawn but the groups need {sum(n for _, n in groups)}")
    rng.shuffle(chosen)
    grouped, k = [], 0
    for group, n in groups:
        grouped += [(group, *pick) for pick in chosen[k:k + n]]
        k += n
    # A second shuffle before numbering, so each annotator's sets mix shared and own-group ones.
    rng.shuffle(grouped)
    width = max(2, len(str(len(grouped))))
    sets = []
    for k, (group, cid, model, a) in enumerate(grouped, start=1):
        c = collections[cid]
        sets.append(PairSet(f"B{k:0{width}d}", group, cid, model, c.dataset, c.level, c.entity, a,
                            texts[(cid, model, a)], texts[(cid, model, other_prompt(a))]))
    return sets


def write_key(sets: Sequence[PairSet], path: str | Path, seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "sets": [asdict(s) for s in sets],
    }, indent=1, ensure_ascii=False), encoding="utf-8")


def read_key(path: str | Path) -> list[PairSet]:
    return [PairSet(**s) for s in json.loads(Path(path).read_text(encoding="utf-8"))["sets"]]


def _item(set_id: str, order: int, collection: Collection, summary_a: str, summary_b: str) -> dict:
    pos = [t for t, p in zip(collection.texts, collection.polarities) if p == "positive"]
    neg = [t for t, p in zip(collection.texts, collection.polarities) if p == "negative"]
    rule = RATING_RULE.get(collection.dataset, ("a positive rating", "a negative rating"))
    return {"id": set_id, "order": order, "topic": collection.topic,
            "minority": "negative" if len(neg) < len(pos) else "positive",
            "summaries": {"A": summary_a, "B": summary_b},
            "reviews": {"positive": pos, "negative": neg},
            "rating_rule": {"positive": rule[0], "negative": rule[1]}}


def web_items(sets: Sequence[PairSet], collections: Mapping[str, Collection]) -> list[dict]:
    """The sets for the labelling page. As blind as Part A: no model, prompt, level or collection
    id. The group says only which annotators label the set."""
    return [{**_item(s.set_id, n, collections[s.collection_id], s.summary_a, s.summary_b), "group": s.group}
            for n, s in enumerate(sets, start=1)]


def practice_item(collection: Collection, summary_a: str, summary_b: str) -> dict:
    return {**_item("P1", 0, collection, summary_a, summary_b), "practice": True}


# --- labels -----------------------------------------------------------------------------------

def _value(raw: object, allowed: Sequence[str]) -> str | None:
    v = str(raw).strip().lower() if isinstance(raw, str) else ""
    return v if v in allowed else None


def web_labels(docs: Iterable[object]) -> dict[str, dict]:
    """{set_id: {"revealed": bool, "A": answers, "B": answers}} from the page's per-set
    documents ({"id": ..., "labels": {...}}). Other people's input: anything malformed is
    skipped, and a value outside a question's answers becomes None."""
    out: dict[str, dict] = {}
    for doc in docs:
        if not isinstance(doc, Mapping) or not isinstance(doc.get("id"), str):
            continue
        labels = doc.get("labels")
        if not isinstance(labels, Mapping):
            continue
        row: dict = {"revealed": labels.get("revealed") is True}
        for slot in SLOTS:
            answers = labels.get(slot) if isinstance(labels.get(slot), Mapping) else {}
            row[slot] = {q: _value(answers.get(q), allowed) for q, allowed in QUESTIONS.items()}
            note = answers.get("note")
            row[slot]["note"] = note.strip() if isinstance(note, str) and note.strip() else None
        out[doc["id"]] = row
    return out


def is_complete(row: Mapping | None) -> bool:
    return bool(row) and row.get("revealed") is True and all(
        row[slot][q] is not None for slot in SLOTS for q in QUESTIONS)


def assigned_sets(sets: Sequence[PairSet], groups: Iterable[str] | None) -> list[PairSet]:
    if groups is None:
        return list(sets)
    wanted = set(groups)
    return [s for s in sets if s.group in wanted]


# --- codes ------------------------------------------------------------------------------------

def strength(answer: str, majority: str) -> int:
    """Majority-side strength of a perceived split: 2 "almost all" on the majority side, 1
    "mostly", 0 "evenly split", -1 and -2 the same on the minority side."""
    k = Q2_VALUES.index(answer) - 2
    return k if majority == "positive" else -k


def q2_class(answer: str, level: str) -> str:
    s, t = strength(answer, majority_side(level)), true_strength(level)
    return "more balanced" if s < t else "about right" if s == t else "more one-sided"


def human_ma(answer: str, level: str, coding: str = "a9") -> float:
    """Perceived majority share minus the true majority share."""
    p = CODINGS[coding][answer]
    share = p if majority_side(level) == "positive" else 1.0 - p
    return share - true_majority_share(level)


def ordinal_alpha(ratings: Mapping[str, Sequence[object]], order: Sequence[object]) -> float:
    """Krippendorff's alpha with the ordinal metric: the distance between two answers grows with
    how many pooled answers lie between them (Krippendorff 2004, section 11.4)."""
    units = [[v for v in vals if v is not None] for vals in ratings.values()]
    pooled = Counter(v for u in units if len(u) >= 2 for v in u)
    rank = {v: i for i, v in enumerate(order)}

    def distance(x: object, y: object) -> float:
        i, j = sorted((rank[x], rank[y]))
        between = sum(pooled[order[g]] for g in range(i, j + 1))
        return (between - (pooled[x] + pooled[y]) / 2) ** 2

    return krippendorff_alpha_distance(ratings, distance)


def auc(scores: Sequence[float], positives: Sequence[bool]) -> float:
    """Probability that a positive case scores higher than a negative one; ties count half."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return math.nan
    ranks = st.rankdata(np.r_[pos, neg])
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _interval(values: Sequence[float]) -> tuple[float, float]:
    finite = [v for v in values if v == v]
    if not finite:
        return math.nan, math.nan
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def _cluster_interval(rows: Sequence[Mapping], key: str, stat: Callable[[list], float],
                      n_boot: int, seed: int) -> tuple[float, float]:
    by: dict[object, list] = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    keys = sorted(by)
    if not keys:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    return _interval([stat([r for k in rng.integers(0, len(keys), len(keys)) for r in by[keys[k]]])
                      for _ in range(n_boot)])


def _f(x: object) -> float | None:
    """JSON-friendly number: numpy scalars become floats, NaN becomes None."""
    if x is None:
        return None
    x = float(x)
    return None if x != x else x


# --- analysis ---------------------------------------------------------------------------------

def judgments(sets: Sequence[PairSet], labels: Mapping[str, Mapping[str, Mapping]],
              scores: Mapping[tuple[str, str, str], Mapping]) -> list[dict]:
    """One row per (annotator, set, summary), with the answers and the automatic scores of that
    summary."""
    rows = []
    for who, by_set in labels.items():
        for s in sets:
            row = by_set.get(s.set_id)
            if not row:
                continue
            for slot in SLOTS:
                cond = s.condition(slot)
                auto = scores.get((s.collection_id, s.model, cond), {})
                rows.append({"annotator": who, "set_id": s.set_id, "group": s.group, "slot": slot,
                             "unit": f"{s.set_id}-{slot}", "condition": cond, "model": s.model,
                             "dataset": s.dataset, "level": s.level,
                             "entity": f"{s.dataset}:{s.entity}", "majority": majority_side(s.level),
                             **{q: row[slot][q] for q in QUESTIONS},
                             **{f: auto.get(f) for f in SCORE_FIELDS}})
    return rows


def _summary_means(rows: Sequence[Mapping], value: Callable[[Mapping], float | None]) -> dict:
    acc: dict[str, list] = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            acc[r["unit"]].append(v)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def _paired(sets: Sequence[PairSet], means: Mapping[str, float]) -> tuple[list[float], list[str]]:
    """Per set, `kw_strat` minus `direct`, with the set's entity."""
    diffs, entities = [], []
    for s in sets:
        kw, d = means.get(f"{s.set_id}-{s.slot('kw_strat')}"), means.get(f"{s.set_id}-{s.slot('direct')}")
        if kw is not None and d is not None:
            diffs.append(kw - d)
            entities.append(f"{s.dataset}:{s.entity}")
    return diffs, entities


def _spearman(xs: Sequence[float | None], ys: Sequence[float | None]) -> dict:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return {"rho": None, "p": None, "n": len(pairs)}  # a constant side has no correlation
    res = st.spearmanr(*zip(*pairs))
    return {"rho": _f(res.statistic), "p": _f(res.pvalue), "n": len(pairs)}


def _alpha(rows: Sequence[Mapping], question: str, annotators: Sequence[str],
           units: Iterable[str] | None = None) -> float:
    want = set(units) if units is not None else None
    ratings: dict[str, list] = defaultdict(lambda: [None] * len(annotators))
    index = {a: i for i, a in enumerate(annotators)}
    for r in rows:
        if r["annotator"] in index and (want is None or r["unit"] in want):
            ratings[r["unit"]][index[r["annotator"]]] = r[question]
    if sum(sum(v is not None for v in vals) >= 2 for vals in ratings.values()) == 0:
        return math.nan
    try:
        if question == "q3":
            return krippendorff_alpha(ratings)
        return ordinal_alpha(ratings, QUESTIONS[question])
    except ValueError:
        return math.nan


def _alpha_interval(rows: Sequence[Mapping], question: str, annotators: Sequence[str],
                    n_boot: int, seed: int) -> tuple[float, float]:
    """Resample sets; a set drawn twice counts twice."""
    by_set: dict[str, list] = defaultdict(list)
    for r in rows:
        by_set[r["set_id"]].append(r)
    keys = sorted(by_set)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        draw = [{**r, "unit": f"{r['unit']}#{n}"}
                for n, k in enumerate(rng.integers(0, len(keys), len(keys))) for r in by_set[keys[k]]]
        values.append(_alpha(draw, question, annotators))
    return _interval(values)


def analyse(sets: Sequence[PairSet], labels: Mapping[str, Mapping[str, Mapping]],
            scores: Mapping[tuple[str, str, str], Mapping],
            assign: Mapping[str, Sequence[str]] | None = None, part_a: Mapping | None = None,
            n_boot: int = 5000, seed: int = 0) -> dict:
    """Every registered Part B analysis (docs/design/2026-10-30-kw-partb-analysis.md, section 4)
    and the reading of the result (section 5). Only annotators who finished every set assigned
    to them are analysed."""
    included, excluded = [], {}
    for who, by_set in sorted(labels.items()):
        mine = assigned_sets(sets, None if assign is None else assign.get(who, ()))
        done = sum(is_complete(by_set.get(s.set_id)) for s in mine)
        if mine and done == len(mine):
            included.append(who)
        else:
            excluded[who] = f"{done} of {len(mine)}"
    use = {who: {sid: row for sid, row in labels[who].items() if is_complete(row)} for who in included}
    rows = judgments(sets, use, scores)
    report: dict = {"annotators": included, "excluded": excluded, "n_sets": len(sets),
                    "n_judgments": len(rows)}
    if not rows:
        report["checks"] = checks(report, part_a)
        return report

    q1 = _summary_means(rows, lambda r: Q1_CODE[r["q1"]])
    diffs, entities = _paired(sets, q1)
    lo, hi = cluster_bootstrap_ci(np.array(diffs), entities, n_boot=n_boot, seed=seed)
    units = sorted(q1)
    first = {r["unit"]: r for r in rows}
    report["q1"] = {
        "n_sets": len(diffs), "mean_diff": _f(np.mean(diffs)), "ci_lo": _f(lo), "ci_hi": _f(hi),
        "p_wilcoxon": _f(wilcoxon_p(np.array(diffs), "greater")),
        "mean": {p: _f(np.mean([q1[u] for u in units if first[u]["condition"] == p])) for p in PROMPTS},
        "spearman": {"registered": _spearman([q1[u] for u in units], [first[u]["recall_minority"] for u in units]),
                     "deattributed": _spearman([q1[u] for u in units],
                                               [first[u]["recall_minority_deattr"] for u in units])},
    }

    strengths = _summary_means(rows, lambda r: strength(r["q2"], r["majority"]))
    sdiffs, _ = _paired(sets, strengths)
    kw_balanced, d_balanced = sum(d < 0 for d in sdiffs), sum(d > 0 for d in sdiffs)
    n = kw_balanced + d_balanced
    shares = {}
    for p in PROMPTS:
        for side in ("positive", "negative"):
            got = Counter(q2_class(r["q2"], r["level"]) for r in rows
                          if r["condition"] == p and r["majority"] == side)
            total = sum(got.values())
            shares[f"{p}|{side}"] = {c: _f(got[c] / total) if total else None
                                     for c in ("more balanced", "about right", "more one-sided")} | {"n": total}
    report["q2"] = {
        "classes": shares,
        "sign_test": {"kw_strat_more_balanced": kw_balanced, "direct_more_balanced": d_balanced,
                      "ties": len(sdiffs) - n,
                      "p": _f(st.binomtest(kw_balanced, n, 0.5).pvalue) if n else None},
        "human_ma": {},
    }
    for coding in CODINGS:
        hm = _summary_means(rows, lambda r, c=coding: human_ma(r["q2"], r["level"], c))
        by_cell = {}
        for p in PROMPTS:
            for side in ("positive", "negative"):
                cell = [{"v": hm[u], "entity": first[u]["entity"]} for u in hm
                        if first[u]["condition"] == p and first[u]["majority"] == side]
                vals = [c["v"] for c in cell]
                clo, chi = _cluster_interval(cell, "entity", lambda d: float(np.mean([x["v"] for x in d])),
                                             n_boot, seed)
                by_cell[f"{p}|{side}"] = {"mean": _f(np.mean(vals)) if vals else None,
                                          "ci_lo": _f(clo), "ci_hi": _f(chi), "n": len(vals)}
        against = {}
        for name, field in (("registered", "majority_amplification"),
                            ("polarity_only", "majority_amplification_polarity_only")):
            pairs = [(hm[u], first[u][field]) for u in hm if first[u][field] is not None]
            h, c = map(np.array, zip(*pairs)) if pairs else (np.array([]), np.array([]))
            correlated = len(pairs) >= 3 and np.ptp(h) > 0 and np.ptp(c) > 0  # a constant side has none
            against[name] = {"pearson": _f(st.pearsonr(h, c).statistic) if correlated else None,
                             "spearman": _f(st.spearmanr(h, c).statistic) if correlated else None,
                             "mean_bias": _f(np.mean(h - c)) if pairs else None, "n": len(pairs)}
        report["q2"]["human_ma"][coding] = {"by_cell": by_cell, "vs_classifier": against}

    report["q3"] = {"yes_share": {p: _f(np.mean([r["q3"] == "yes" for r in rows if r["condition"] == p]))
                                  for p in PROMPTS}, "auc": {}}
    for name, field in (("registered", "unsupported_rate"), ("deattributed", "unsupported_rate_deattr")):
        cases = [r for r in rows if r[field] is not None]
        stat = lambda d, f=field: auc([r[f] for r in d], [r["q3"] == "yes" for r in d])
        clo, chi = _cluster_interval(cases, "entity", stat, n_boot, seed)
        report["q3"]["auc"][name] = {"auc": _f(stat(cases)), "ci_lo": _f(clo), "ci_hi": _f(chi), "n": len(cases)}

    report["q4"] = {p: {v: _f(np.mean([r["q4"] == v for r in rows if r["condition"] == p])) for v in Q4_VALUES}
                    for p in PROMPTS}

    agreement = {}
    shared_units = {r["unit"] for r in rows if r["group"] == "shared"}
    for q in QUESTIONS:
        lo, hi = _alpha_interval(rows, q, included, n_boot=min(n_boot, 2000), seed=seed)
        agreement[q] = {"alpha": _f(_alpha(rows, q, included)), "ci_lo": _f(lo), "ci_hi": _f(hi),
                        "metric": "nominal" if q == "q3" else "ordinal",
                        "shared_sets": _f(_alpha(rows, q, included, shared_units)) if len(included) > 2 else None,
                        "pairs": {f"{a} vs {b}": _f(_alpha(rows, q, [a, b])) for a, b in combinations(included, 2)}}
    report["agreement"] = agreement
    report["checks"] = checks(report, part_a)
    return report


def checks(report: Mapping, part_a: Mapping | None = None) -> dict:
    """What the paper needs from the human evaluation (amendment 10, section 3), each decided as
    fixed in docs/design/2026-10-30-kw-partb-analysis.md, and the framing that follows
    (amendment 9, section 6). A question whose alpha is below 0.667 cannot satisfy an item."""
    agreement = report.get("agreement", {})

    def alpha_ok(q: str) -> bool:
        a = (agreement.get(q) or {}).get("alpha")
        return a is not None and a >= ALPHA_MIN

    def above(x: object, bound: float) -> bool:
        return x is not None and x > bound

    i = ((part_a or {}).get("primary_faithfulness") or {}).get("choice") == "de-attributed"
    q1 = report.get("q1") or {}
    ii = above(q1.get("mean_diff"), 0) and above(q1.get("ci_lo"), 0) and alpha_ok("q1")
    sign = (report.get("q2") or {}).get("sign_test") or {}
    iii = (sign.get("p") is not None and sign["p"] < 0.05
           and sign["kw_strat_more_balanced"] > sign["direct_more_balanced"] and alpha_ok("q2"))
    rho = ((q1.get("spearman") or {}).get("registered") or {}).get("rho")
    iv = rho is not None and rho >= 0.3
    primary = "deattributed" if i else "registered"
    auc_ci = ((report.get("q3") or {}).get("auc") or {}).get(primary) or {}
    v = above(auc_ci.get("ci_lo"), 0.5) and alpha_ok("q3")
    return {"i_part_a": i, "ii_q1_coverage": ii, "iii_q2_balance": iii, "iv_coverage_agreement": iv,
            "v_q3_unsupported": v, "q3_score": primary,
            "framing": "trade-off" if ii and iv else "measurement"}
