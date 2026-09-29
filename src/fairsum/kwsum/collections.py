"""Imbalanced 8-review collections.

A collection is 8 reviews of one entity with a fixed positive:negative split.
Fixing the split, instead of taking whatever an entity's reviews happen to be,
is what lets the study treat imbalance as a manipulated variable (spec §3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LEVELS: dict[str, tuple[int, int]] = {
    "bal": (4, 4),
    "pos75": (6, 2),
    "pos88": (7, 1),
    "neg75": (2, 6),
    "neg88": (1, 7),
}
SKEWED_LEVELS: tuple[str, ...] = ("pos75", "pos88", "neg75", "neg88")


@dataclass(frozen=True)
class Collection:
    collection_id: str
    dataset: str
    split: str
    level: str
    entity: str
    topic: str
    review_ids: tuple[str, ...]
    texts: tuple[str, ...]
    polarities: tuple[str, ...]

    @property
    def n_pos(self) -> int:
        return sum(p == "positive" for p in self.polarities)

    @property
    def n_neg(self) -> int:
        return sum(p == "negative" for p in self.polarities)

    @property
    def majority(self) -> str | None:
        if self.n_pos == self.n_neg:
            return None
        return "positive" if self.n_pos > self.n_neg else "negative"

    @property
    def minority(self) -> str | None:
        if self.majority is None:
            return None
        return "negative" if self.majority == "positive" else "positive"

    @property
    def source_neg_share(self) -> float:
        return self.n_neg / len(self.polarities)

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Collection":
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


class InsufficientEntitiesError(ValueError):
    """Too few entities can supply a level's split. Never silently build fewer."""


def split_entities(entities: Sequence[str], dev_fraction: float, seed: int) -> dict[str, str]:
    """Assign every entity to dev or test once, so the two never share one."""
    ordered = sorted(set(entities))
    perm = np.random.default_rng(seed).permutation(len(ordered))
    n_dev = int(round(dev_fraction * len(ordered)))
    return {ordered[j]: ("dev" if rank < n_dev else "test") for rank, j in enumerate(perm)}


def _eligible(pool: pd.DataFrame, n_pos: int, n_neg: int) -> list[str]:
    counts = pool.groupby(["entity", "polarity"]).size().unstack(fill_value=0)
    pos = counts["positive"] if "positive" in counts.columns else 0
    neg = counts["negative"] if "negative" in counts.columns else 0
    return sorted(counts.index[(pos >= n_pos) & (neg >= n_neg)])


def _sample(
    reviews: pd.DataFrame,
    collection_id: str,
    dataset: str,
    split: str,
    level: str,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
) -> Collection:
    reviews = reviews.sort_values("review_id")
    pos = reviews[reviews["polarity"] == "positive"]
    neg = reviews[reviews["polarity"] == "negative"]
    picked = pd.concat([
        pos.iloc[rng.choice(len(pos), size=n_pos, replace=False)],
        neg.iloc[rng.choice(len(neg), size=n_neg, replace=False)],
    ])
    picked = picked.iloc[rng.permutation(len(picked))]
    return Collection(
        collection_id=collection_id,
        dataset=dataset,
        split=split,
        level=level,
        entity=str(picked["entity"].iloc[0]),
        topic=str(picked["topic"].iloc[0]),
        review_ids=tuple(picked["review_id"]),
        texts=tuple(picked["text"]),
        polarities=tuple(picked["polarity"]),
    )


def build_collections(
    pool: pd.DataFrame,
    dataset: str,
    n_test: int,
    n_dev: int,
    levels: Sequence[str] = tuple(LEVELS),
    dev_fraction: float = 0.2,
    seed: int = 42,
) -> list[Collection]:
    """Collections for every level and split. Entities are sampled at random.

    v1 picked drugs alphabetically (vault note `F4 Design Confounds`). Here an
    entity is used at most once per level and split, and review order within a
    collection is shuffled, so the minority review is not always last.
    """
    rng = np.random.default_rng(seed)
    assignment = split_entities(pool["entity"].unique(), dev_fraction, seed)
    by_entity = {entity: group for entity, group in pool.groupby("entity")}

    out: list[Collection] = []
    for level in levels:
        n_pos, n_neg = LEVELS[level]
        eligible = _eligible(pool, n_pos, n_neg)
        for split, wanted in (("test", n_test), ("dev", n_dev)):
            candidates = [e for e in eligible if assignment[e] == split]
            if len(candidates) < wanted:
                raise InsufficientEntitiesError(
                    f"{dataset}/{level}/{split}: need {wanted} entities with >= {n_pos} "
                    f"positive and >= {n_neg} negative usable reviews, found {len(candidates)}"
                )
            chosen = rng.choice(len(candidates), size=wanted, replace=False)
            for i, idx in enumerate(chosen):
                out.append(_sample(
                    by_entity[candidates[idx]],
                    f"{dataset}-{split}-{level}-{i:03d}",
                    dataset, split, level, n_pos, n_neg, rng,
                ))
    return out


def write_collections(collections: Iterable[Collection], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in collections:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_collections(path: str | Path) -> list[Collection]:
    with open(path, encoding="utf-8") as fh:
        return [Collection.from_dict(json.loads(line)) for line in fh if line.strip()]
