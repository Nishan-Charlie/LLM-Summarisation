"""Run configuration (TOML) for the keyword-prompting study.

Paths are anchored to the repo root, so scripts and notebooks agree on where
data and results live regardless of the working directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from fairsum.kwsum.collections import LEVELS
from fairsum.kwsum.prompts import CONDITIONS

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "kw"
POOLS_DIR = DATA_DIR / "pools"
COLLECTIONS_PATH = DATA_DIR / "collections.jsonl"
KEYWORDS_PATH = DATA_DIR / "keywords.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

KNOWN_DATASETS = ("amazon", "drugs")
KNOWN_BACKENDS = ("hf", "openai", "anthropic")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    model_id: str
    load_in_4bit: bool = False
    batch_size: int | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_workers: int = 8


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    datasets: tuple[str, ...]
    levels: tuple[str, ...]
    n_test: int
    n_dev: int
    dev_fraction: float
    k: int
    diversity: float
    embedder: str
    conditions: tuple[str, ...]
    max_new_tokens: int
    repetition_penalty: float
    batch_size: int
    nli_model: str
    sentiment_model: str
    threshold: float
    models: tuple[ModelSpec, ...]
    #: 3-class model used only to decide neutral vs opinion; None = sentiment_model alone.
    neutral_model: str | None = None

    def model(self, name: str) -> ModelSpec:
        for spec in self.models:
            if spec.name == name:
                return spec
        raise KeyError(f"no model named {name!r}; known: {[m.name for m in self.models]}")

    @property
    def results_dir(self) -> Path:
        return REPO_ROOT / "results" / "kw" / self.run_id

    @property
    def generations_path(self) -> Path:
        return self.results_dir / "generations.jsonl"

    @property
    def scores_path(self) -> Path:
        return self.results_dir / "scores.jsonl"


def _check(values, known, what: str) -> tuple:
    unknown = [v for v in values if v not in known]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; known: {list(known)}")
    return tuple(values)


def load_config(path: str | Path) -> RunConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    col, kw, gen, sc = raw["collections"], raw["keywords"], raw["generation"], raw["scoring"]
    models = tuple(ModelSpec(**m) for m in raw.get("models", []))
    _check([m.backend for m in models], KNOWN_BACKENDS, "backend")
    return RunConfig(
        run_id=raw["run_id"],
        seed=int(raw["seed"]),
        datasets=_check(col["datasets"], KNOWN_DATASETS, "dataset"),
        levels=_check(col["levels"], LEVELS, "level"),
        n_test=int(col["n_test"]),
        n_dev=int(col["n_dev"]),
        dev_fraction=float(col["dev_fraction"]),
        k=int(kw["k"]),
        diversity=float(kw["diversity"]),
        embedder=kw["embedder"],
        conditions=_check(gen["conditions"], CONDITIONS, "condition"),
        max_new_tokens=int(gen["max_new_tokens"]),
        repetition_penalty=float(gen["repetition_penalty"]),
        batch_size=int(gen["batch_size"]),
        nli_model=sc["nli_model"],
        sentiment_model=sc["sentiment_model"],
        threshold=float(sc["threshold"]),
        models=models,
        neutral_model=sc.get("neutral_model"),
    )
