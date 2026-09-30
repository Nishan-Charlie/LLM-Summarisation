"""Metrics from the NLI matrix and sentiment labels (spec §5).

Model calls are replaced by fixed matrices and fakes, so every number here can
be checked by hand.
"""

import numpy as np
import pytest

from fairsum.kwsum.collections import Collection
from fairsum.kwsum.scoring import (
    combine_labels,
    count_error,
    coverage_recalls,
    invented_numbers,
    keyword_uptake,
    labels_from_probs,
    majority_amplification,
    matrix_metrics,
    SCORER_VERSION,
    score_generation,
    summary_neg_share,
    validate_sentiment,
)

POS75 = ["positive"] * 6 + ["negative"] * 2
KW = {
    "global": ["sleep"],
    "strat": {"positive": ["sleep"], "negative": ["headaches"]},
    "strat_neg": {"positive": ["sleep"], "negative": ["headaches"]},
}


def coll(polarities, texts=None, level="pos75"):
    n = len(polarities)
    return Collection(
        "c1", "drugs", "test", level, "E", "E",
        tuple(f"r{i}" for i in range(n)),
        tuple(texts or [f"review {i}" for i in range(n)]),
        tuple(polarities),
    )


class TestBuildSentiment:
    """The instrument the scorer and the human check both use (spec §5)."""

    @pytest.fixture
    def fake_hf(self, monkeypatch):
        import fairsum.kwsum.scoring as scoring

        class FakeHF:
            def __init__(self, model_name, device=None, neutral_band=0.8):
                self.args = (model_name, device, neutral_band)

        monkeypatch.setattr(scoring, "HFSentiment", FakeHF)
        return scoring

    def test_hybrid_when_a_neutral_model_is_configured(self, fake_hf):
        s = fake_hf.build_sentiment("siebert", "cardiff", device="cpu")
        assert isinstance(s, fake_hf.HybridSentiment)
        assert s.gate.args == ("cardiff", "cpu", 0.8)
        assert s.polarity.args == ("siebert", "cpu", 0.0)

    def test_single_model_without_a_neutral_model(self, fake_hf):
        assert fake_hf.build_sentiment("siebert").args == ("siebert", None, 0.8)


class TestMatrixMetrics:
    def test_unsupported_and_support(self):
        entail = np.array([[0.9, 0.1, 0.2], [0.1, 0.3, 0.8]])  # 2 reviews x 3 props
        m = matrix_metrics(entail, np.zeros_like(entail), threshold=0.5)
        assert m["unsupported_rate"] == pytest.approx(1 / 3)
        assert m["support_score"] == pytest.approx((0.9 + 0.3 + 0.8) / 3)

    def test_contradiction_counts_only_unsupported_props(self):
        entail = np.array([[0.9, 0.1], [0.1, 0.2]])
        contra = np.array([[0.6, 0.7], [0.0, 0.1]])
        assert matrix_metrics(entail, contra, 0.5)["contradiction_rate"] == pytest.approx(0.5)


class TestCoverage:
    def test_recalls_gap_and_attribution(self):
        entail = np.zeros((8, 2))
        entail[:6, 0] = 0.9  # every majority review supports prop 0
        entail[6, 1] = 0.9   # one minority review supports prop 1
        r = coverage_recalls(entail, POS75, majority="positive", threshold=0.5)
        assert r["recall_majority"] == 1.0 and r["recall_minority"] == 0.5
        assert r["coverage_gap"] == pytest.approx(0.5)
        assert r["attribution_share_minority"] == pytest.approx(0.5)

    def test_balanced_has_no_gap(self):
        r = coverage_recalls(np.full((8, 1), 0.9), ["positive", "negative"] * 4,
                             majority=None, threshold=0.5)
        assert r["coverage_gap"] is None and r["recall_pos"] == 1.0


class TestProportions:
    def test_neg_share_ignores_neutral(self):
        assert summary_neg_share(["positive", "neutral", "negative", "negative"]) == pytest.approx(2 / 3)

    def test_no_polar_props_is_none_not_zero(self):
        assert summary_neg_share(["neutral"]) is None

    def test_majority_amplification_sign(self):
        assert majority_amplification(0.0, coll(POS75)) == pytest.approx(0.25)
        assert majority_amplification(0.5, coll(POS75)) == pytest.approx(-0.25)
        assert majority_amplification(1.0, coll(["negative"] * 6 + ["positive"] * 2)) == pytest.approx(0.25)

    def test_balanced_has_no_amplification(self):
        assert majority_amplification(0.3, coll(["positive", "negative"] * 4)) is None


class TestCounts:
    def test_error_against_true_counts(self):
        c = coll(POS75)
        assert count_error([(6, "positive"), (2, "negative")], c) == (0.0, 0)
        assert count_error([(4, "positive"), (4, "negative")], c) == (2.0, 1)
        assert count_error([], c) == (None, None)

    def test_counts_larger_than_the_collection_are_not_count_claims(self):
        # "100 Capsules received positive ..." must not become an error of 94
        assert count_error([(100, "positive"), (6, "positive")], coll(POS75)) == (0.0, 0)
        assert count_error([(32, "negative")], coll(POS75)) == (None, None)


class TestLexical:
    def test_invented_numbers(self):
        assert invented_numbers(["Took 20 mg daily", "Worked in 3 days"], ["I take 20 mg every morning"]) == 1

    def test_keyword_uptake_handles_plurals(self):
        uptake = keyword_uptake("The side effect was mild and sleep improved",
                                ["side effects", "sleep", "weight gain"])
        assert uptake == pytest.approx(2 / 3)

    def test_uptake_is_none_without_keywords(self):
        assert keyword_uptake("anything", []) is None


class FakeNLI:
    """Entails when the proposition's first word appears in the review."""

    def matrices(self, premises, hypotheses):
        e = np.array([[0.9 if h.split()[0].lower() in p.lower() else 0.1 for h in hypotheses]
                      for p in premises])
        return e, np.zeros_like(e)


class FakeSentiment:
    def labels(self, texts):
        return ["negative" if "bad" in t.lower() else "positive" for t in texts]


class NeutralSentiment:
    def labels(self, texts):
        return ["neutral"] * len(texts)


def gen(raw, status="ok", condition="kw_strat"):
    return {"collection_id": "c1", "dataset": "drugs", "split": "test", "level": "pos75",
            "condition": condition, "model": "m", "raw": raw, "status": status}


TEXTS = ["Sleep improved a lot"] * 6 + ["Headaches were bad", "Nausea was bad"]
SUMMARY = "Sleep improved for most users. Headaches were bad for some."


class TestScoreGeneration:
    def test_an_unscored_status_carries_no_metrics(self):
        row = score_generation(gen("", status="empty"), coll(POS75), KW, FakeNLI(), FakeSentiment())
        assert row["status"] == "empty"
        assert row["unsupported_rate"] is None and row["delta_p"] is None

    def test_end_to_end_row(self):
        row = score_generation(gen(SUMMARY), coll(POS75, TEXTS), KW, FakeNLI(), FakeSentiment())
        assert row["n_props"] == 2
        assert row["unsupported_rate"] == 0.0
        assert row["summary_neg_share"] == pytest.approx(0.5)
        assert row["delta_p"] == pytest.approx(0.25)
        assert row["majority_amplification"] == pytest.approx(-0.25)
        assert row["recall_minority"] == pytest.approx(0.5)
        assert row["keyword_uptake"] == pytest.approx(1.0)
        assert row["uer"] == pytest.approx(0.125)  # binary: UER = |dp| / 2

    def test_rows_carry_the_scorer_version_and_generation_status(self):
        row = score_generation(gen(SUMMARY), coll(POS75, TEXTS), KW, FakeNLI(), FakeSentiment())
        assert row["scorer_version"] == SCORER_VERSION and row["gen_status"] == "ok"

    def test_status_is_recomputed_at_scoring_time(self):
        g = {**gen("Out of 8 reviews, 6 are positive and 2 are negative."),
             "completion_tokens": 20, "stop_reason": None}
        row = score_generation(g, coll(POS75, TEXTS), KW, FakeNLI(), FakeSentiment(), max_new_tokens=256)
        assert row["gen_status"] == "ok" and row["status"] == "no_summary"
        assert row["unsupported_rate"] is None

    def test_a_truncated_summary_loses_its_cut_off_tail(self):
        g = {**gen("Sleep improved for most users. Headaches were bad for some. Nausea was"),
             "completion_tokens": 256, "stop_reason": None}
        row = score_generation(g, coll(POS75, TEXTS), KW, FakeNLI(), FakeSentiment(), max_new_tokens=256)
        assert row["status"] == "truncated" and row["n_props"] == 2

    def test_negation_manipulation_flag(self):
        changed = {**KW, "strat_neg": {"positive": ["sleep"], "negative": ["no headaches"]}}
        assert score_generation(gen(SUMMARY), coll(POS75, TEXTS), changed, FakeNLI(),
                                FakeSentiment())["negation_changed"] is True
        assert score_generation(gen(SUMMARY), coll(POS75, TEXTS), KW, FakeNLI(),
                                FakeSentiment())["negation_changed"] is False

    def test_all_neutral_summary_has_no_proportion_metrics(self):
        row = score_generation(gen(SUMMARY), coll(POS75, TEXTS), KW, FakeNLI(), NeutralSentiment())
        assert row["delta_p"] is None and row["majority_amplification"] is None
        assert row["fairsum_status"] == "degenerate_summary" and row["uer"] is None
        assert row["unsupported_rate"] == 0.0  # faithfulness is still measured


class TestLabelsFromProbs:
    def test_three_class_models_take_the_argmax(self):
        probs = np.array([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])
        assert labels_from_probs(probs, ["negative", "neutral", "positive"], neutral_band=0.9) == [
            "neutral", "negative"]

    def test_binary_models_call_low_confidence_neutral(self):
        probs = np.array([[0.05, 0.95], [0.3, 0.7], [0.99, 0.01]])
        assert labels_from_probs(probs, ["negative", "positive"], neutral_band=0.8) == [
            "positive", "neutral", "negative"]

    def test_no_band_means_pure_argmax(self):
        probs = np.array([[0.45, 0.55]])
        assert labels_from_probs(probs, ["negative", "positive"], neutral_band=0.0) == ["positive"]


def test_hybrid_labels_take_neutral_from_the_gate_and_polarity_from_the_polarity_model():
    gate = ["neutral", "positive", "negative", "neutral"]
    polarity = ["positive", "negative", "negative", "negative"]
    assert combine_labels(gate, polarity) == ["neutral", "negative", "negative", "neutral"]


class BothInstruments:
    """Hybrid calls the minority clause neutral; the polarity model alone calls it negative."""

    def labels(self, texts):
        return self.labels_both(texts)[0]

    def labels_both(self, texts):
        polarity = ["negative" if "bad" in t.lower() else "positive" for t in texts]
        hybrid = ["neutral" if "bad" in t.lower() else "positive" for t in texts]
        return hybrid, polarity


def test_rows_carry_the_polarity_only_robustness_variant():
    row = score_generation(gen(SUMMARY), coll(POS75, TEXTS), KW, FakeNLI(), BothInstruments())
    assert row["summary_neg_share"] == pytest.approx(0.0)          # hybrid: 1 positive, 1 neutral
    assert row["summary_neg_share_polarity_only"] == pytest.approx(0.5)
    assert row["delta_p_polarity_only"] == pytest.approx(0.25)
    assert row["majority_amplification_polarity_only"] == pytest.approx(-0.25)


def test_validate_sentiment_reports_accuracy():
    c = coll(POS75, TEXTS)
    report = validate_sentiment(FakeSentiment(), [c], n=8, seed=0)
    assert report["accuracy"] == pytest.approx(1.0) and report["n"] == 8


class GateNeutralisesNegatives:
    """A neutral gate that fires on half the negative reviews and no positive one."""

    def labels(self, texts):
        return ["neutral" if t == "Nausea was bad" else ("negative" if "bad" in t else "positive")
                for t in texts]


def test_validation_reports_neutral_rate_by_true_polarity():
    report = validate_sentiment(GateNeutralisesNegatives(), [coll(POS75, TEXTS)], n=8, seed=0)
    assert report["neutral_rate_by_polarity"] == {"negative": 0.5, "positive": 0.0}
    assert report["polar_accuracy"] == pytest.approx(1.0)  # every non-neutral call is right
    assert report["accuracy"] == pytest.approx(7 / 8)
