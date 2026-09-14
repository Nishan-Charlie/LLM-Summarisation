"""Building summarization instances, and splitting drugs for calibration.

Annotation is per review and one-time; instance construction is combinatorial
and free. That asymmetry is what makes a calibration set of a few hundred
instances affordable from a pool of ~720 hand-annotated reviews.

Because instances within a drug share reviews, they are not independent. The
splits are therefore **drug-disjoint**: exchangeability is argued at the drug
level, where it is defensible. See the vault note `Calibration Design`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_SPLIT_FRACTIONS = {"dev": 0.2, "calibration": 0.4, "test": 0.4}

# An instance is `ae_singleton` if at least this many events are reported by
# exactly one review -- the regime H1 lives in.
SINGLETON_THRESHOLD = 3
# ... and `ae_dense` if it carries at least this many distinct events.
DENSE_THRESHOLD = 5


@dataclass
class Instance:
    """One summarization problem: N reviews of a single drug for one indication."""

    instance_id: str
    product_id: str
    # None when the instance spans several indications, which is the common case
    # under `group_by="drug"`. Prompts then make no indication claim at all,
    # rather than asserting the first review's condition as v1 did.
    condition_name: str | None
    reviews: list[dict[str, Any]]
    ae_condition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def review_ids(self) -> list[str]:
        return [r["review_id"] for r in self.reviews]

    @property
    def texts(self) -> list[str]:
        return [r["text"] for r in self.reviews]


def assign_drug_splits(
    drugs: Iterable[str],
    seed: int,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
) -> dict[str, str]:
    """Assign each drug to exactly one split, disjointly and reproducibly.

    Sorted before shuffling so the result depends on the seed and the drug set,
    never on the order the caller happened to supply.
    """
    unique = sorted(set(drugs))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique))

    names = list(fractions)
    n = len(shuffled)
    # Largest-remainder allocation, so the counts sum to n exactly.
    exact = {name: fractions[name] * n for name in names}
    sizes = {name: int(np.floor(exact[name])) for name in names}
    for name in sorted(names, key=lambda k: exact[k] - sizes[k], reverse=True):
        if sum(sizes.values()) >= n:
            break
        sizes[name] += 1

    assignment: dict[str, str] = {}
    cursor = 0
    for name in names:
        for drug in shuffled[cursor : cursor + sizes[name]]:
            assignment[drug] = name
        cursor += sizes[name]
    for drug in shuffled[cursor:]:  # any remainder from rounding
        assignment[drug] = names[-1]
    return assignment


def build_instances(
    reviews: pd.DataFrame,
    n_reviews: int = 8,
    seed: int = 0,
    max_instances: int | None = None,
    drugs: set[str] | None = None,
    instances_per_group: int = 1,
    group_by: str = "drug",
    product_col: str = "product_id",
    condition_col: str = "condition_name",
    id_col: str = "review_id",
) -> list[Instance]:
    """Sample instances of `n_reviews` reviews of the same drug.

    Groups are visited in a seeded random order rather than sorted order, which
    stops the sample from being the alphabetical prefix of the drug space
    (`F4 Design Confounds` #1).

    `group_by="drug"` (default) puts all of a drug's reviews in one pool and
    leaves `condition_name` unset, so the prompt makes no indication claim.
    `group_by="drug_indication"` pins one indication per instance, which is
    stricter but costs most of the pool: on real Druglib only 42 (drug,
    indication) groups reach 8 reviews, against 144 drug groups.
    """
    if group_by not in ("drug", "drug_indication"):
        raise ValueError(
            f"group_by must be 'drug' or 'drug_indication', got {group_by!r}"
        )

    frame = reviews if drugs is None else reviews[reviews[product_col].isin(drugs)]

    if group_by == "drug_indication":
        keys = sorted(
            {(str(p), str(c)) for p, c in zip(frame[product_col], frame[condition_col])}
        )
    else:
        keys = sorted({(str(p), None) for p in frame[product_col]})

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))

    instances: list[Instance] = []
    for idx in order:
        if max_instances is not None and len(instances) >= max_instances:
            break
        product_id, condition_name = keys[idx]

        group = frame[frame[product_col] == product_id]
        if condition_name is not None:
            group = group[group[condition_col] == condition_name]
        if len(group) < n_reviews:
            continue

        records = group.to_dict("records")
        for draw in range(instances_per_group):
            if max_instances is not None and len(instances) >= max_instances:
                break
            picked_idx = rng.choice(len(records), size=n_reviews, replace=False)
            picked = [records[i] for i in picked_idx]
            indications = sorted(
                {str(r[condition_col]) for r in picked if condition_col in r}
            )
            instances.append(
                Instance(
                    instance_id=_instance_id(
                        product_id, condition_name or "*", picked, id_col
                    ),
                    product_id=product_id,
                    condition_name=condition_name,
                    reviews=picked,
                    metadata={
                        "seed": seed,
                        "draw": draw,
                        "group_by": group_by,
                        "indications": indications,
                    },
                )
            )
    return instances


def _instance_id(
    product_id: str,
    condition_name: str,
    reviews: Sequence[Mapping[str, Any]],
    id_col: str,
) -> str:
    """Content-addressed id, so the same review set always gets the same name."""
    payload = "|".join(
        [product_id, condition_name, *sorted(str(r[id_col]) for r in reviews)]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def instance_adverse_events(
    per_review_events: Sequence[set[str]],
    with_counts: bool = False,
) -> set[str] | tuple[set[str], dict[str, int]]:
    """Union the adverse events reported across an instance's reviews.

    With `with_counts`, also return how many reviews report each event -- the
    `n_i` that H1 predicts the inclusion rate `p_i` depends on.
    """
    counts = Counter()
    for events in per_review_events:
        counts.update(set(events))
    union = set(counts)
    if with_counts:
        return union, dict(counts)
    return union


def classify_ae_condition(per_review_events: Sequence[set[str]]) -> str:
    """Label an instance by the shape of its adverse-event distribution.

    Replaces v1's rating-polarity conditions, which manipulated a different
    construct from the one being measured (`F4 Design Confounds` #3).
    """
    _, counts = instance_adverse_events(per_review_events, with_counts=True)
    singletons = sum(1 for n in counts.values() if n == 1)
    if singletons >= SINGLETON_THRESHOLD:
        return "ae_singleton"
    if len(counts) >= DENSE_THRESHOLD:
        return "ae_dense"
    return "ae_sparse"


def select_annotation_batch(
    reviews: pd.DataFrame,
    batch_size: int,
    n_drugs: int,
    n_reviews: int = 8,
    seed: int = 0,
    exclude_splits: Iterable[str] = ("test",),
    split_col: str = "split",
    product_col: str = "product_id",
    id_col: str = "review_id",
) -> pd.DataFrame:
    """Draw reviews to annotate, spread evenly across drugs that can form an instance.

    Only drugs with at least `n_reviews` reviews are eligible -- the rule
    `build_instances` applies -- so every label is reusable in the bulk set. The
    first pilot drew from any drug, and half its reviews came from drugs no
    instance can ever use (`tests/test_pilot_batch.py`).

    Raises rather than shrinking the batch or relaxing eligibility: a quietly
    smaller or skewed batch is what went unnoticed the first time.
    """
    excluded = set(exclude_splits)
    candidates = reviews[~reviews[split_col].isin(excluded)] if excluded else reviews
    sizes = candidates[product_col].value_counts()
    eligible = sorted(sizes[sizes >= n_reviews].index)
    if len(eligible) < n_drugs:
        raise ValueError(
            f"only {len(eligible)} eligible drugs (>= {n_reviews} reviews, not in "
            f"{sorted(excluded)}), but {n_drugs} were requested"
        )

    rng = np.random.default_rng(seed)
    chosen = [eligible[i] for i in rng.permutation(len(eligible))[:n_drugs]]
    return _round_robin(candidates, chosen, batch_size, rng, product_col, id_col)


def select_agreement_subset(
    batch: pd.DataFrame,
    size: int,
    seed: int = 0,
    product_col: str = "product_id",
    id_col: str = "review_id",
) -> pd.DataFrame:
    """Pick the reviews a second annotator labels, spread across the batch's drugs.

    Agreement measured mostly on one drug's vocabulary would say little about
    the schema, so the subset is spread the same way the batch is.
    """
    rng = np.random.default_rng(seed)
    drugs = sorted(batch[product_col].unique())
    order = [drugs[i] for i in rng.permutation(len(drugs))]
    return _round_robin(batch, order, size, rng, product_col, id_col)


def _round_robin(
    frame: pd.DataFrame,
    drugs: Sequence[str],
    size: int,
    rng: np.random.Generator,
    product_col: str,
    id_col: str,
) -> pd.DataFrame:
    """Take one review per drug per round, in `drugs` order, until `size` rows.

    Each drug's reviews are sorted by id before the seeded shuffle, so the draw
    depends on the seed and the review set, never on row order.
    """
    queues: dict[str, list[Any]] = {}
    for drug in drugs:
        group = frame[frame[product_col] == drug].sort_values(id_col)
        queues[drug] = list(group.index[rng.permutation(len(group))])

    available = sum(len(queue) for queue in queues.values())
    if size > available:
        raise ValueError(
            f"cannot fill {size} reviews from {len(drugs)} drugs holding {available}"
        )

    picked: list[Any] = []
    while len(picked) < size:
        for drug in drugs:
            if queues[drug] and len(picked) < size:
                picked.append(queues[drug].pop())
    return frame.loc[picked].reset_index(drop=True)
