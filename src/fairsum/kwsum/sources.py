"""Review pools for the keyword-prompting study.

A pool is one row per usable review: the entity it is about (a product or a
drug), a topic string for the prompt, the cleaned text, the rating and a binary
polarity. Every rule that decides whether a review is usable lives here, so
collection building never meets a review it would have to reject.

REFER (Huang et al., NewSum 2025) keeps reviews of 30-120 words; so do we.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd

MIN_WORDS = 30
MAX_WORDS = 120
POOL_COLUMNS = ["review_id", "dataset", "entity", "topic", "text", "rating", "polarity", "n_words"]

AMAZON_REPO = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_CATEGORY = "Health_and_Personal_Care"
DRUGS_REPO = "lewtun/drug-reviews"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MAX_TOPIC_WORDS = 20


def clean_text(raw: str) -> str:
    """Unescape HTML entities, drop tags, strip one pair of wrapping quotes.

    Drugs.com text is escaped twice in places (`&amp;#039;`), so unescaping
    repeats until the text stops changing.
    """
    text = raw or ""
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def amazon_polarity(stars: float) -> str | None:
    """4-5 stars positive, 1-2 negative, 3 dropped as neither."""
    if stars >= 4:
        return "positive"
    if stars <= 2:
        return "negative"
    return None


def drugs_polarity(rating: float) -> str | None:
    """Drugs.com's 1-10 scale: 7 and up positive, 4 and below negative."""
    if rating >= 7:
        return "positive"
    if rating <= 4:
        return "negative"
    return None


def _row(
    review_id: str,
    dataset: str,
    entity: str,
    topic: str,
    text: str,
    rating: float,
    polarity: str | None,
) -> dict | None:
    n_words = len(text.split())
    if polarity is None or not MIN_WORDS <= n_words <= MAX_WORDS:
        return None
    return {
        "review_id": review_id,
        "dataset": dataset,
        "entity": entity,
        "topic": topic,
        "text": text,
        "rating": rating,
        "polarity": polarity,
        "n_words": n_words,
    }


def amazon_rows(
    records: Iterable[dict], titles: dict[str, str], fallback_topic: str
) -> Iterator[dict]:
    """Usable Amazon reviews. The review's line position is its id."""
    for i, rec in enumerate(records):
        rating = float(rec["rating"])
        entity = rec["parent_asin"]
        row = _row(
            f"amazon:{i}",
            "amazon",
            entity,
            titles.get(entity, fallback_topic),
            clean_text(rec.get("text", "")),
            rating,
            amazon_polarity(rating),
        )
        if row is not None:
            yield row


def drugs_rows(records: Iterable[dict]) -> Iterator[dict]:
    """Usable Drugs.com reviews. The drug name is both entity and topic."""
    for rec in records:
        rating = float(rec["rating"])
        name = clean_text(rec["drugName"])
        row = _row(
            f"drugs:{rec['Unnamed: 0']}",
            "drugs",
            name,
            name,
            clean_text(rec.get("review", "")),
            rating,
            drugs_polarity(rating),
        )
        if row is not None:
            yield row


def build_pool(rows: Iterable[dict]) -> pd.DataFrame:
    """Rows to a pool, dropping exact duplicate texts (first occurrence wins).

    Drugs.com files some reviews under both a brand and a generic name. Keeping
    both would let one review sit in two entities' collections.
    """
    frame = pd.DataFrame(list(rows), columns=POOL_COLUMNS)
    frame = frame.drop_duplicates(subset="text", keep="first")
    return frame.reset_index(drop=True)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def hub_file(repo_id: str, filename: str) -> Path:
    """Download (or reuse the cached copy of) one file from a Hub dataset repo.

    The raw JSONL is read directly: `datasets` 4.x no longer runs the Amazon
    repo's loading script.
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def load_amazon_titles(meta_path: str | Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    for rec in iter_jsonl(meta_path):
        title = clean_text(rec.get("title") or "")
        if title:
            titles[rec["parent_asin"]] = " ".join(title.split()[:_MAX_TOPIC_WORDS])
    return titles


def build_amazon_pool(category: str = AMAZON_CATEGORY) -> pd.DataFrame:
    reviews = hub_file(AMAZON_REPO, f"raw/review_categories/{category}.jsonl")
    meta = hub_file(AMAZON_REPO, f"raw/meta_categories/meta_{category}.jsonl")
    fallback = f"a {category.replace('_', ' ')} product"
    return build_pool(amazon_rows(iter_jsonl(reviews), load_amazon_titles(meta), fallback))


def build_drugs_pool() -> pd.DataFrame:
    rows: list[dict] = []
    for split in ("train.jsonl", "test.jsonl"):
        rows.extend(drugs_rows(iter_jsonl(hub_file(DRUGS_REPO, split))))
    return build_pool(rows)


POOL_BUILDERS: dict[str, Callable[[], pd.DataFrame]] = {
    "amazon": build_amazon_pool,
    "drugs": build_drugs_pool,
}
