"""Part B of the human evaluation (amendments 9 and 10; docs/design/2026-10-30-kw-partb-analysis.md).

The sample, the blinding, the label parsing, the codes and the reading of the result are checked
with hand-built collections, scores and labels.
"""

import json
import math
from collections import Counter

import pytest

from fairsum.annotation import krippendorff_alpha
from fairsum.kwsum.collections import LEVELS, SKEWED_LEVELS, Collection
from fairsum.kwsum.human_partb import (
    PRACTICE_COLLECTION, PROMPTS, analyse, auc, checks, eligible_pairs, human_ma,
    is_complete, ordinal_alpha, practice_item, q2_class, read_key, sample_sets, strength,
    web_items, web_labels, write_key,
)
from fairsum.kwsum.scoring import SCORER_VERSION

MODELS = ("m1", "m2", "m3", "m4", "m5")
PER_CELL = 14


def coll(dataset: str, level: str, k: int) -> Collection:
    n_pos, n_neg = LEVELS[level]
    pols = ("positive",) * n_pos + ("negative",) * n_neg
    cid = f"{dataset}-test-{level}-{k:03d}"
    return Collection(cid, dataset, "test", level, f"E{k % 9}", f"Product {dataset[0]}{k}",
                      tuple(f"{cid}-r{j}" for j in range(8)),
                      tuple(f"review {j}: it was {'good' if p == 'positive' else 'bad'}" for j, p in enumerate(pols)),
                      pols)


@pytest.fixture(scope="module")
def world():
    collections = {}
    for dataset in ("amazon", "drugs"):
        for level in (*SKEWED_LEVELS, "bal"):
            for k in range(PER_CELL):
                c = coll(dataset, level, k)
                collections[c.collection_id] = c
    collections[PRACTICE_COLLECTION] = Collection(
        PRACTICE_COLLECTION, "drugs", "test", "pos88", "EP", "Practice drug",
        tuple(f"p{j}" for j in range(8)), tuple(f"practice review {j}" for j in range(8)),
        ("positive",) * 7 + ("negative",))
    scores, gens = [], []
    for cid, c in collections.items():
        for m in MODELS:
            for p in PROMPTS:
                status = "ok"
                if cid.endswith("-001") and m == "m1" and p == "kw_strat":
                    status = "refusal"  # m1's pair on every -001 collection is not eligible
                scores.append({"collection_id": cid, "dataset": c.dataset, "split": "test", "level": c.level,
                               "condition": p, "model": m, "status": status, "scorer_version": SCORER_VERSION,
                               "recall_minority": 0.2 if p == "direct" else 0.6,
                               "recall_minority_deattr": 0.3 if p == "direct" else 0.7,
                               "unsupported_rate": 0.5, "unsupported_rate_deattr": 0.4,
                               "majority_amplification": -0.05 if p == "direct" else -0.15,
                               "majority_amplification_polarity_only": -0.05})
                gens.append({"collection_id": cid, "model": m, "condition": p, "split": "test",
                             "raw": f"summary of {cid} by {m} under {p}"})
    return collections, scores, gens


def draw(world, **kw):
    collections, scores, gens = world
    return sample_sets(scores, gens, collections, **kw)


def test_the_sample_has_the_registered_shape(world):
    sets = draw(world, exclude=["amazon-test-pos75-000"])
    assert len(sets) == 80
    assert len({s.collection_id for s in sets}) == 80
    assert Counter((s.dataset, s.level) for s in sets) == {(d, lv): 10 for d in ("amazon", "drugs") for lv in SKEWED_LEVELS}
    assert set(Counter((s.dataset, s.level, s.model) for s in sets).values()) == {2}
    assert Counter(s.group for s in sets) == {"shared": 10, "g1": 35, "g2": 35}
    assert [s.set_id for s in sets] == [f"B{k:02d}" for k in range(1, 81)]
    # Each annotator's sets, in id order, mix the shared ones with their own group's.
    order = [s.group for s in sets if s.group in ("shared", "g1")]
    assert order.index("g1") < 10 and max(i for i, g in enumerate(order) if g == "shared") > 10
    assert all(s.level in SKEWED_LEVELS for s in sets)
    assert "amazon-test-pos75-000" not in {s.collection_id for s in sets}
    assert PRACTICE_COLLECTION not in {s.collection_id for s in sets}


def test_only_pairs_with_two_scored_summaries_are_drawn(world):
    _, scores, _ = world
    ok = eligible_pairs(scores)
    assert ("amazon-test-pos75-001", "m1") not in ok and ("amazon-test-pos75-001", "m2") in ok
    assert not any(s.model == "m1" and s.collection_id.endswith("-001") for s in draw(world))
    old = [{**r, "scorer_version": "0"} for r in scores]
    assert eligible_pairs(old) == set()  # only the current scorer's rows count


def test_the_draw_is_seeded_and_the_order_of_a_and_b_varies(world):
    a, b = draw(world), draw(world)
    assert a == b
    assert draw(world, seed=7) != a
    assert 20 < sum(s.a_condition == "direct" for s in a) < 60
    s = a[0]
    assert {s.condition("A"), s.condition("B")} == set(PROMPTS)
    assert s.summary_a.endswith(s.a_condition) and s.summary_b.endswith(s.condition("B"))
    assert s.slot(s.a_condition) == "A"


def test_too_few_eligible_collections_is_an_error(world):
    collections, scores, gens = world
    few = {cid: c for cid, c in collections.items() if not cid.startswith("drugs-test-neg88") or cid.endswith(("-000", "-002"))}
    with pytest.raises(ValueError, match="drugs neg88"):
        sample_sets(scores, gens, few)


def test_key_round_trip(world, tmp_path):
    sets = draw(world)
    write_key(sets, tmp_path / "key.json", 2026)
    assert read_key(tmp_path / "key.json") == sets


def test_the_page_items_are_blind(world):
    collections, _, _ = world
    sets = draw(world)
    items = web_items(sets, collections)
    # The synthetic summaries name their model and prompt, so they are left out of this check.
    text = json.dumps([{k: v for k, v in it.items() if k != "summaries"} for it in items])
    for word in ("direct", "kw_strat", "pos75", "pos88", "neg75", "neg88", "-test-", "m1", "m5", "E1"):
        assert word not in text
    first = items[0]
    assert set(first) == {"id", "order", "topic", "minority", "summaries", "reviews", "rating_rule", "group"}
    c = collections[sets[0].collection_id]
    assert len(first["reviews"]["positive"]) == c.n_pos and len(first["reviews"]["negative"]) == c.n_neg
    assert first["minority"] == c.minority
    assert first["summaries"] == {"A": sets[0].summary_a, "B": sets[0].summary_b}
    practice = practice_item(collections[PRACTICE_COLLECTION], "a", "b")
    assert practice["practice"] is True and practice["id"] == "P1" and "group" not in practice


def test_labels_from_the_page_are_normalised_and_bad_values_dropped():
    docs = [{"id": "B01", "labels": {"revealed": True,
                                     "A": {"q1": "Most", "q2": "mostly_positive", "q3": "no", "q4": "clear", "note": " x "},
                                     "B": {"q1": "none", "q2": "evenly_split", "q3": "maybe", "q4": "clear"}}},
            {"id": "B02", "labels": "nonsense"}, {"labels": {}}, "junk"]
    out = web_labels(docs)
    assert list(out) == ["B01"]
    assert out["B01"]["A"] == {"q1": "most", "q2": "mostly_positive", "q3": "no", "q4": "clear", "note": "x"}
    assert out["B01"]["B"]["q3"] is None and out["B01"]["B"]["note"] is None
    assert not is_complete(out["B01"])
    out["B01"]["B"]["q3"] = "yes"
    assert is_complete(out["B01"])
    out["B01"]["revealed"] = False
    assert not is_complete(out["B01"])


def test_perceived_split_against_the_true_split():
    assert strength("almost_all_positive", "positive") == 2 and strength("almost_all_positive", "negative") == -2
    assert q2_class("almost_all_positive", "pos88") == "about right"
    assert q2_class("mostly_positive", "pos88") == "more balanced"
    assert q2_class("mostly_negative", "pos75") == "more balanced"   # the minority side
    assert q2_class("almost_all_negative", "neg75") == "more one-sided"
    assert q2_class("mostly_negative", "neg75") == "about right"
    assert human_ma("mostly_positive", "pos88") == pytest.approx(0.75 - 0.875)
    assert human_ma("mostly_negative", "neg75") == pytest.approx(0.0)
    assert human_ma("almost_all_negative", "neg88", "alt") == pytest.approx(0.0)
    assert human_ma("evenly_split", "pos75") == pytest.approx(-0.25)


def test_ordinal_alpha():
    order = ("none", "some", "most")
    same = {f"u{k}": [v, v] for k, v in enumerate(["none", "some", "most"] * 4)}
    assert ordinal_alpha(same, order) == pytest.approx(1.0)
    binary = {f"u{k}": pair for k, pair in enumerate([["yes", "yes"], ["no", "no"], ["yes", "no"], ["no", "no"], ["yes", "yes"]])}
    assert ordinal_alpha(binary, ("no", "yes")) == pytest.approx(krippendorff_alpha(binary))
    base = [["none", "none"], ["most", "most"], ["some", "some"]] * 3
    near = {f"u{k}": p for k, p in enumerate(base + [["none", "some"], ["some", "most"]])}
    far = {f"u{k}": p for k, p in enumerate(base + [["none", "most"], ["most", "none"]])}
    assert ordinal_alpha(near, order) > ordinal_alpha(far, order)


def test_auc():
    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auc([0.5, 0.5], [True, False]) == 0.5
    assert auc([0.1, 0.9], [True, False]) == 0.0
    assert math.isnan(auc([0.1], [True]))


def labels_for(sets, kw_q1="most", d_q1="none", flip=None, q2_kw="evenly_split"):
    """Every set labelled: `kw_strat` gets kw_q1, `direct` d_q1; `flip` swaps them on some sets."""
    out = {}
    for n, s in enumerate(sets):
        row = {"revealed": True}
        for slot in ("A", "B"):
            kw = s.condition(slot) == "kw_strat"
            q1 = kw_q1 if kw else d_q1
            if flip and flip(n):
                q1 = d_q1 if kw else kw_q1
            maj = "positive" if s.level.startswith("pos") else "negative"
            q2 = q2_kw if kw else f"almost_all_{maj}"
            row[slot] = {"q1": q1, "q2": q2, "q3": "yes" if n % 3 == 0 else "no", "q4": "clear", "note": None}
        out[s.set_id] = row
    return out


def score_map(world):
    _, scores, _ = world
    return {(r["collection_id"], r["model"], r["condition"]): r for r in scores}


def test_a_clear_coverage_gain_passes_and_sets_the_trade_off_framing(world):
    sets = draw(world)
    labels = {"ann1": labels_for(sets), "ann2": labels_for(sets)}
    rep = analyse(sets, labels, score_map(world), part_a={"primary_faithfulness": {"choice": "de-attributed"}},
                  n_boot=300)
    assert rep["annotators"] == ["ann1", "ann2"] and rep["excluded"] == {}
    assert rep["q1"]["mean_diff"] == pytest.approx(1.0) and rep["q1"]["ci_lo"] > 0
    assert rep["q1"]["p_wilcoxon"] < 0.001
    assert rep["q1"]["spearman"]["registered"]["rho"] > 0.9
    assert rep["agreement"]["q1"]["alpha"] == pytest.approx(1.0)
    sign = rep["q2"]["sign_test"]
    assert sign["kw_strat_more_balanced"] == 80 and sign["p"] < 0.001
    assert rep["q2"]["classes"]["kw_strat|positive"]["more balanced"] == 1.0
    assert rep["q2"]["classes"]["direct|negative"]["about right"] == pytest.approx(
        sum(s.level == "neg88" for s in sets) / sum(s.level.startswith("neg") for s in sets))
    c = rep["checks"]
    assert c["i_part_a"] and c["ii_q1_coverage"] and c["iii_q2_balance"] and c["iv_coverage_agreement"]
    assert c["q3_score"] == "deattributed" and c["framing"] == "trade-off"
    # Q3 never varies with the automatic score here, so it cannot pass.
    assert rep["q3"]["auc"]["deattributed"]["auc"] == pytest.approx(0.5) and not c["v_q3_unsupported"]


def test_no_gain_or_poor_agreement_means_the_measurement_framing(world):
    sets = draw(world)
    null = {"ann1": labels_for(sets, "some", "some"), "ann2": labels_for(sets, "some", "some")}
    rep = analyse(sets, null, score_map(world), n_boot=200)
    assert rep["q1"]["mean_diff"] == 0 and not rep["checks"]["ii_q1_coverage"]
    assert rep["checks"]["framing"] == "measurement" and not rep["checks"]["i_part_a"]
    # A gain that the two annotators do not agree on (alpha below 0.667) cannot pass item (ii).
    split = {"ann1": labels_for(sets), "ann2": labels_for(sets, flip=lambda n: n % 2 == 0)}
    rep = analyse(sets, split, score_map(world), n_boot=200)
    assert rep["agreement"]["q1"]["alpha"] < 0.667 and not rep["checks"]["ii_q1_coverage"]


def test_only_annotators_who_finished_their_sets_are_analysed(world):
    sets = draw(world)
    full = labels_for(sets)
    half = dict(list(full.items())[:30])
    g1 = {s.set_id: full[s.set_id] for s in sets if s.group in ("g1", "shared")}
    assign = {"ann1": ["g1", "g2", "shared"], "ann2": ["g1", "g2", "shared"], "ann3": ["g1", "shared"]}
    rep = analyse(sets, {"ann1": full, "ann2": half, "ann3": g1}, score_map(world), assign=assign, n_boot=100)
    assert rep["annotators"] == ["ann1", "ann3"]
    assert rep["excluded"] == {"ann2": "30 of 80"}
    assert rep["n_judgments"] == 2 * (80 + 45)


def test_checks_without_any_labels_hold_nothing():
    c = checks({})
    assert not any(c[k] for k in ("i_part_a", "ii_q1_coverage", "iii_q2_balance", "iv_coverage_agreement",
                                  "v_q3_unsupported"))
    assert c["framing"] == "measurement"
