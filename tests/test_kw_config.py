"""Run configuration: the week-1 file loads and bad values fail loudly."""

import pytest

from fairsum.kwsum.config import REPO_ROOT, load_config

CONFIG = REPO_ROOT / "configs" / "kw_week1.toml"


def test_week1_config_loads():
    cfg = load_config(CONFIG)
    assert cfg.n_test == 60 and cfg.n_dev == 10
    assert len(cfg.conditions) == 9
    assert cfg.model("llama3.1-8b").load_in_4bit
    assert cfg.model("llama3.2-3b").batch_size == 8
    assert cfg.sentiment_model == "siebert/sentiment-roberta-large-english"
    assert cfg.neutral_model == "cardiffnlp/twitter-roberta-base-sentiment-latest"


def test_unknown_model_lists_the_known_ones():
    with pytest.raises(KeyError, match="llama3.2-3b"):
        load_config(CONFIG).model("gpt-9")


def test_results_live_under_the_repo():
    cfg = load_config(CONFIG)
    assert cfg.results_dir == REPO_ROOT / "results" / "kw" / "week1"
    assert cfg.generations_path.name == "generations.jsonl"


def _variant(tmp_path, old, new):
    text = CONFIG.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "cfg.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_unknown_condition_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown condition"):
        load_config(_variant(tmp_path, '"direct", "refer"', '"bogus", "refer"'))


def test_unknown_level_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown level"):
        load_config(_variant(tmp_path, '"bal", "pos75"', '"half", "pos75"'))


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown backend"):
        load_config(_variant(tmp_path, 'backend = "hf"', 'backend = "vllm"'))
