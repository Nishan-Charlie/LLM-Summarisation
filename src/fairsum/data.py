"""Loading Druglib.com reviews (UCI ML Repository id 461).

`ucimlrepo` is not installed in this environment, so the direct-zip path the v1
notebook carried as a fallback is the primary loader here. It has no extra
dependency and pins the archive URL explicitly, which is better for
reproducibility anyway.

The pool keeps every review with usable text. v1 discarded mid-scale ratings
because its sampling conditions were built from rating polarity; v2 builds them
from adverse-event density (`F4 Design Confounds` #3), so there is nothing to
gain from throwing the middle of the scale away.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Mapping

import pandas as pd

from fairsum.metrics import EFFECTIVENESS_LABELS

DRUGLIB_URL = (
    "https://archive.ics.uci.edu/static/public/461/"
    "drug+review+dataset+druglib+com.zip"
)
TEXT_FIELDS = ("benefitsReview", "sideEffectsReview", "commentsReview")

MIN_WORDS = 20
MAX_WORDS = 200


def load_druglib_raw(cache_dir: str | Path = "data") -> pd.DataFrame:
    """Download and concatenate the Druglib train/test TSVs, caching locally."""
    cache = Path(cache_dir) / "druglib_raw.csv"
    if cache.exists():
        return pd.read_csv(cache)

    with urllib.request.urlopen(DRUGLIB_URL) as response:
        buffer = io.BytesIO(response.read())

    frames = []
    with zipfile.ZipFile(buffer) as archive:
        for name in archive.namelist():
            if not name.endswith(".tsv"):
                continue
            with archive.open(name) as handle:
                frames.append(
                    pd.read_csv(handle, sep="\t", index_col=0, encoding="latin-1")
                )
    if not frames:
        raise RuntimeError(f"no .tsv members found in {DRUGLIB_URL}")

    raw = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cache, index=False)
    return raw


def druglib_text(row: Mapping) -> str:
    """Join the three free-text fields, skipping the ones that are absent."""
    parts = []
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)


def _review_id(product_id: str, condition: str, text: str) -> str:
    """Content-addressed, so a review keeps its id across rebuilds of the pool."""
    payload = f"{product_id}|{condition}|{text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_review_pool(
    raw: pd.DataFrame,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> pd.DataFrame:
    """Turn raw Druglib rows into the review pool the rest of the pipeline uses.

    Keeps each review's own indication rather than the drug's first one, which
    is what stops a prompt from asserting an indication the reviews are not
    about (`F4 Design Confounds` #5).
    """
    frame = raw.copy()
    frame.columns = [str(c).strip() for c in frame.columns]

    frame["text"] = frame.apply(druglib_text, axis=1)
    frame["word_count"] = frame["text"].str.split().str.len().fillna(0)
    frame["effectiveness"] = frame["effectiveness"].astype("string").str.strip()
    frame["product_id"] = frame["urlDrugName"].astype("string").str.strip()
    frame["condition_name"] = frame["condition"].astype("string").str.strip()

    keep = (
        frame["product_id"].notna()
        & frame["condition_name"].notna()
        & frame["effectiveness"].isin(EFFECTIVENESS_LABELS)
        & frame["word_count"].between(min_words, max_words)
    )
    frame = frame.loc[keep].copy()

    frame["review_id"] = [
        _review_id(p, c, t)
        for p, c, t in zip(frame["product_id"], frame["condition_name"], frame["text"])
    ]
    frame = frame.drop_duplicates(subset="review_id")

    columns = [
        "review_id",
        "product_id",
        "condition_name",
        "text",
        "effectiveness",
        "rating",
        "sideEffects",
        "word_count",
    ]
    return frame[[c for c in columns if c in frame.columns]].reset_index(drop=True)
