"""Scoring one generation (spec §5).

One NLI matrix, with reviews as premises and summary propositions as
hypotheses, yields the faithfulness, contradiction and content-coverage
metrics. A 3-class sentiment classifier over the same propositions yields the
proportion metrics. A generation whose status is not scored still gets a row,
with every metric None: a failure is never a number (FairSum invariant).
"""

from __future__ import annotations

import re
from typing import Mapping, Protocol, Sequence

import numpy as np

from fairsum.kwsum.collections import Collection
from fairsum.kwsum.compliance import SCORED_STATUSES, classify_output
from fairsum.kwsum.prompts import KEYWORD_BASE
from fairsum.kwsum.propositions import split_summary
from fairsum.metrics import MetricStatus, fairsum_metrics

#: Bump whenever a change alters any score. Rows from another version are never
#: reused on resume and are dropped by the analysis (final review, finding I5).
#: v2: count-statement handling, summary-header trimming, truncated tails,
#: status recomputed at scoring time, hybrid sentiment.
SCORER_VERSION = "2"

ID_FIELDS = ("collection_id", "dataset", "split", "level", "condition", "model")
METRIC_FIELDS = (
    "n_props", "n_count_statements", "n_meta", "n_words",
    "count_statements_parsed", "count_abs_error", "count_wrong", "invented_numbers",
    "keyword_uptake", "keyword_uptake_minority",
    "unsupported_rate", "support_score", "contradiction_rate",
    "recall_pos", "recall_neg", "recall_majority", "recall_minority", "coverage_gap",
    "attribution_share_minority",
    "n_polar_props", "summary_neg_share", "source_neg_share", "delta_p",
    "majority_amplification", "fairsum_status", "bur", "uer", "sof", "spd",
)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_WORD = re.compile(r"[a-z0-9]+")


class NLIModel(Protocol):
    def matrices(self, premises: Sequence[str], hypotheses: Sequence[str]) -> tuple[np.ndarray, np.ndarray]: ...


class SentimentModel(Protocol):
    def labels(self, texts: Sequence[str]) -> list[str]: ...


def matrix_metrics(entail: np.ndarray, contra: np.ndarray, threshold: float) -> dict:
    max_e, max_c = entail.max(axis=0), contra.max(axis=0)
    return {
        "unsupported_rate": float(np.mean(max_e < threshold)),
        "support_score": float(max_e.mean()),
        "contradiction_rate": float(np.mean((max_c >= threshold) & (max_e < threshold))),
    }


def coverage_recalls(entail: np.ndarray, polarities: Sequence[str], majority: str | None,
                     threshold: float) -> dict:
    pol = np.asarray(polarities)
    covered = (entail >= threshold).any(axis=1)

    def recall(group: str) -> float | None:
        mask = pol == group
        return float(covered[mask].mean()) if mask.any() else None

    out = {
        "recall_pos": recall("positive"), "recall_neg": recall("negative"),
        "recall_majority": None, "recall_minority": None, "coverage_gap": None,
        "attribution_share_minority": None,
    }
    if majority is None:
        return out
    minority = "negative" if majority == "positive" else "positive"
    out["recall_majority"], out["recall_minority"] = recall(majority), recall(minority)
    out["coverage_gap"] = out["recall_majority"] - out["recall_minority"]
    supported = (entail >= threshold).any(axis=0)
    if supported.any():
        by_minority = (entail[pol == minority] >= threshold).any(axis=0)
        out["attribution_share_minority"] = float((by_minority & supported).sum() / supported.sum())
    return out


def summary_neg_share(labels: Sequence[str]) -> float | None:
    polar = [label for label in labels if label in ("positive", "negative")]
    if not polar:
        return None
    return polar.count("negative") / len(polar)


def majority_amplification(summary_neg: float, collection: Collection) -> float | None:
    """Summary's majority share minus the source's. > 0: the majority is amplified."""
    if collection.majority is None:
        return None
    delta_neg = summary_neg - collection.source_neg_share
    return delta_neg if collection.majority == "negative" else -delta_neg


def count_error(counts: Sequence[tuple[int, str]], collection: Collection) -> tuple[float | None, int | None]:
    # A number larger than the collection is not a count of its reviews (e.g. "100 Capsules").
    counts = [(n, p) for n, p in counts if 0 <= n <= len(collection.texts)]
    if not counts:
        return None, None
    truth = {"positive": collection.n_pos, "negative": collection.n_neg}
    errors = [abs(n - truth[polarity]) for n, polarity in counts]
    return float(np.mean(errors)), int(any(e > 0 for e in errors))


def invented_numbers(propositions: Sequence[str], source_texts: Sequence[str]) -> int:
    source = set(_NUMBER.findall(" ".join(source_texts)))
    return sum(1 for p in propositions for num in _NUMBER.findall(p) if num not in source)


def _norm_tokens(text: str) -> set[str]:
    toks = _WORD.findall(text.lower().replace("’", "'"))
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks}


def keyword_uptake(summary: str, phrases: Sequence[str]) -> float | None:
    if not phrases:
        return None
    have = _norm_tokens(summary)
    return sum(_norm_tokens(p) <= have for p in phrases) / len(phrases)


def _uptake_phrases(condition: str, keywords: Mapping, group: str | None = None) -> list[str]:
    base = KEYWORD_BASE.get(condition)
    if base is None:
        return []
    if base == "global":
        return list(keywords["global"])
    strat = keywords["strat"]
    return list(strat[group]) if group else list(strat["positive"]) + list(strat["negative"])


def _negation_changed(keywords: Mapping) -> bool:
    """Whether kw_strat_neg's prompt differs from kw_strat's for this collection (H3)."""
    return any(list(keywords["strat"][g]) != list(keywords["strat_neg"][g])
               for g in ("positive", "negative"))


def score_generation(gen: Mapping, collection: Collection, keywords: Mapping,
                     nli: NLIModel, sentiment: SentimentModel, threshold: float = 0.5,
                     max_new_tokens: int | None = None) -> dict:
    """One score row. With `max_new_tokens`, status is recomputed from the raw
    output with the current splitter; the generation-time status is kept as
    `gen_status`."""
    raw = gen["raw"]
    status = gen["status"]
    if max_new_tokens is not None:
        status = classify_output(raw, int(gen.get("completion_tokens", 0)), max_new_tokens,
                                 stop_reason=gen.get("stop_reason"))
    row = {k: gen[k] for k in ID_FIELDS}
    row.update(status=status, gen_status=gen["status"], scorer_version=SCORER_VERSION,
               negation_changed=_negation_changed(keywords))
    row.update(dict.fromkeys(METRIC_FIELDS))
    if status not in SCORED_STATUSES:
        return row

    split = split_summary(raw, drop_unterminated_tail=(status == "truncated"))
    props = list(split.propositions)
    err, wrong = count_error(split.counts, collection)
    row.update(
        n_props=len(props), n_count_statements=split.n_count_statements,
        n_meta=split.n_meta, n_words=len(raw.split()),
        count_statements_parsed=len(split.counts), count_abs_error=err, count_wrong=wrong,
        invented_numbers=invented_numbers(props, collection.texts),
        keyword_uptake=keyword_uptake(raw, _uptake_phrases(gen["condition"], keywords)),
        source_neg_share=collection.source_neg_share,
    )
    if collection.minority and KEYWORD_BASE.get(gen["condition"]) == "strat":
        row["keyword_uptake_minority"] = keyword_uptake(
            raw, _uptake_phrases(gen["condition"], keywords, collection.minority))
    if not props:
        return row

    entail, contra = nli.matrices(list(collection.texts), props)
    row.update(matrix_metrics(entail, contra, threshold))
    row.update(coverage_recalls(entail, collection.polarities, collection.majority, threshold))

    labels = sentiment.labels(props)
    row["n_polar_props"] = sum(label in ("positive", "negative") for label in labels)
    neg = summary_neg_share(labels)
    row["summary_neg_share"] = neg
    if neg is not None:
        row["delta_p"] = abs(neg - collection.source_neg_share)
        row["majority_amplification"] = majority_amplification(neg, collection)

    fm = fairsum_metrics(
        {"positive": collection.n_pos, "negative": collection.n_neg},
        {"positive": labels.count("positive"), "negative": labels.count("negative")},
        labels=["positive", "negative"],
    )
    row["fairsum_status"] = fm.status.value
    if fm.status is MetricStatus.OK:
        row.update(bur=fm.bur, uer=fm.uer, sof=fm.sof, spd=fm.spd)
    return row


def validate_sentiment(sentiment: SentimentModel, collections: Sequence[Collection],
                       n: int = 400, seed: int = 0) -> dict:
    """Classifier accuracy on source reviews against their star polarity (spec §5)."""
    reviews = [(c.dataset, t, p) for c in collections for t, p in zip(c.texts, c.polarities)]
    idx = np.random.default_rng(seed).choice(len(reviews), size=min(n, len(reviews)), replace=False)
    sample = [reviews[i] for i in idx]
    labels = sentiment.labels([t for _, t, _ in sample])
    truth = [p for _, _, p in sample]
    correct = [label == p for label, p in zip(labels, truth)]
    by_dataset: dict[str, float] = {}
    for ds in sorted({d for d, _, _ in sample}):
        hits = [ok for ok, (d, _, _) in zip(correct, sample) if d == ds]
        by_dataset[ds] = float(np.mean(hits))
    polar = [(label, p) for label, p in zip(labels, truth) if label != "neutral"]
    # A neutral gate that fires more on one polarity biases |dp| (final review, I3).
    neutral_by_polarity = {
        pol: float(np.mean([label == "neutral" for label, p in zip(labels, truth) if p == pol]))
        for pol in sorted(set(truth))
    }
    accuracy = float(np.mean(correct))
    polar_accuracy = float(np.mean([label == p for label, p in polar])) if polar else float("nan")
    return {
        "n": len(sample), "accuracy": accuracy, "polar_accuracy": polar_accuracy,
        "neutral_rate": float(np.mean([label == "neutral" for label in labels])),
        "neutral_rate_by_polarity": neutral_by_polarity,
        "by_dataset": by_dataset, "passes": polar_accuracy >= 0.90,
    }


def labels_from_probs(probs: np.ndarray, names: Sequence[str], neutral_band: float = 0.0) -> list[str]:
    """Class probabilities to positive / neutral / negative labels.

    A 3-class model keeps its argmax. A binary model has no neutral class, so a
    prediction whose confidence falls below `neutral_band` is called neutral:
    summaries contain factual statements ("comes in a pack of three") that a
    binary classifier would otherwise force onto one side.
    """
    has_neutral = "neutral" in names
    out: list[str] = []
    for row in np.asarray(probs, dtype=float):
        label = names[int(row.argmax())]
        if not has_neutral and row.max() < neutral_band:
            label = "neutral"
        out.append(label)
    return out


def combine_labels(gate: Sequence[str], polarity: Sequence[str]) -> list[str]:
    """Neutral from the 3-class gate, polarity from the binary model otherwise.

    The review-trained binary model is the better polarity judge (0.91 against
    star ratings, vs 0.76 for the 3-class model), but it is ~0.99 confident on
    neutral facts too ("it comes in a pack of three"), so neutrality comes from
    the 3-class model.
    """
    return ["neutral" if g == "neutral" else p for g, p in zip(gate, polarity)]


class HybridSentiment:
    """A 3-class neutral gate in front of a binary polarity model."""

    def __init__(self, gate: "HFSentiment", polarity: "HFSentiment"):
        self.gate = gate
        self.polarity = polarity

    def labels(self, texts: Sequence[str]) -> list[str]:
        return combine_labels(self.gate.labels(texts), self.polarity.labels(texts))


def _normalise_label(name: str) -> str:
    name = name.lower()
    if name.startswith("neg"):
        return "negative"
    if name.startswith("pos"):
        return "positive"
    return "neutral"


class HFNLI:
    """Pairwise NLI probabilities. fp16 on GPU; reviews are premises."""

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli",
                 device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        names = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        self.entail_idx = next(i for i, n in enumerate(names) if "entail" in n)
        self.contra_idx = next(i for i, n in enumerate(names) if "contra" in n)

    def matrices(self, premises: Sequence[str], hypotheses: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        pairs = [(p, h) for p in premises for h in hypotheses]
        chunks = []
        for s in range(0, len(pairs), self.batch_size):
            batch = pairs[s : s + self.batch_size]
            enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch], return_tensors="pt",
                                 padding=True, truncation=True, max_length=512).to(self.device)
            with self._torch.no_grad():
                chunks.append(self.model(**enc).logits.float().softmax(-1).cpu().numpy())
        probs = np.concatenate(chunks).reshape(len(premises), len(hypotheses), -1)
        return probs[..., self.entail_idx], probs[..., self.contra_idx]


class HFSentiment:
    """Sentiment labels normalised to positive / neutral / negative.

    Works with 3-class models and with binary ones; for binary models
    `neutral_band` turns low-confidence predictions into neutral.
    """

    def __init__(self, model_name: str = "siebert/sentiment-roberta-large-english",
                 device: str | None = None, batch_size: int = 64, neutral_band: float = 0.8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.neutral_band = neutral_band
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.names = [_normalise_label(self.model.config.id2label[i])
                      for i in range(self.model.config.num_labels)]

    def labels(self, texts: Sequence[str]) -> list[str]:
        out: list[str] = []
        for s in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(list(texts[s : s + self.batch_size]), return_tensors="pt",
                                 padding=True, truncation=True, max_length=256).to(self.device)
            with self._torch.no_grad():
                probs = self.model(**enc).logits.float().softmax(-1).cpu().numpy()
            out.extend(labels_from_probs(probs, self.names, self.neutral_band))
        return out
