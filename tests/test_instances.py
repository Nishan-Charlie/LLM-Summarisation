"""Instance construction and drug-disjoint splits.

Two v1 confounds are pinned shut here as regressions (vault note
`F4 Design Confounds`):

  #1  `groupby("product_id")` + `break` returned the alphabetical prefix of the
      drug space, so the sample was chosen by drug name.
  #5  `condition_name` was taken from the first review of a drug, so prompts
      could state an indication the shown reviews were not about.

The split invariant matters for a different reason: conformal calibration needs
exchangeability, and instances built from a shared review pool are not
independent. Drug-disjoint splits are what make the argument defensible.
"""

import pandas as pd
import pytest

from fairsum.instances import (
    assign_drug_splits,
    build_instances,
    classify_ae_condition,
    instance_adverse_events,
)


def reviews_frame(n_drugs=26, per_drug=12, conditions_per_drug=1):
    """Synthetic review pool: drugs named a, b, c ... with enough reviews each."""
    rows = []
    for i in range(n_drugs):
        drug = chr(ord("a") + i) * 3
        for c in range(conditions_per_drug):
            for j in range(per_drug):
                rows.append(
                    {
                        "review_id": f"{drug}-{c}-{j}",
                        "product_id": drug,
                        "condition_name": f"cond{c}",
                        "text": f"review {j} of {drug}",
                        "effectiveness": "Highly Effective",
                    }
                )
    return pd.DataFrame(rows)


class TestAssignDrugSplits:
    def test_every_drug_lands_in_exactly_one_split(self):
        drugs = [f"d{i}" for i in range(30)]
        assignment = assign_drug_splits(drugs, seed=0)
        assert set(assignment) == set(drugs)
        assert set(assignment.values()) <= {"dev", "calibration", "test"}

    def test_splits_are_disjoint_by_drug(self):
        drugs = [f"d{i}" for i in range(30)]
        assignment = assign_drug_splits(drugs, seed=0)
        buckets = {"dev": set(), "calibration": set(), "test": set()}
        for drug, split in assignment.items():
            buckets[split].add(drug)
        assert buckets["dev"] & buckets["calibration"] == set()
        assert buckets["dev"] & buckets["test"] == set()
        assert buckets["calibration"] & buckets["test"] == set()

    def test_is_deterministic_for_a_given_seed(self):
        drugs = [f"d{i}" for i in range(30)]
        assert assign_drug_splits(drugs, seed=7) == assign_drug_splits(drugs, seed=7)

    def test_different_seeds_give_different_assignments(self):
        drugs = [f"d{i}" for i in range(30)]
        assert assign_drug_splits(drugs, seed=1) != assign_drug_splits(drugs, seed=2)

    def test_respects_the_requested_proportions(self):
        drugs = [f"d{i}" for i in range(100)]
        assignment = assign_drug_splits(
            drugs, seed=0, fractions={"dev": 0.2, "calibration": 0.4, "test": 0.4}
        )
        counts = pd.Series(list(assignment.values())).value_counts()
        assert counts["dev"] == 20
        assert counts["calibration"] == 40
        assert counts["test"] == 40

    def test_assignment_does_not_depend_on_input_order(self):
        drugs = [f"d{i}" for i in range(30)]
        assert assign_drug_splits(drugs, seed=3) == assign_drug_splits(
            list(reversed(drugs)), seed=3
        )


class TestBuildInstances:
    def test_each_instance_has_the_requested_number_of_reviews(self):
        instances = build_instances(reviews_frame(), n_reviews=8, seed=0, max_instances=10)
        assert all(len(inst.reviews) == 8 for inst in instances)

    def test_an_instance_never_mixes_drugs(self):
        for inst in build_instances(reviews_frame(), n_reviews=8, seed=0, max_instances=10):
            assert {r["product_id"] for r in inst.reviews} == {inst.product_id}

    def test_an_instance_never_mixes_indications_under_indication_grouping(self):
        # F4 #5: v1 labelled an instance with the drug's first condition, which
        # could differ from the conditions of the reviews actually shown. Under
        # `group_by="drug"` the fix is to claim no indication at all -- see
        # test_grouping.py; here the stricter mode must pin exactly one.
        frame = reviews_frame(conditions_per_drug=3)
        for inst in build_instances(
            frame, n_reviews=8, seed=0, max_instances=10, group_by="drug_indication"
        ):
            assert {r["condition_name"] for r in inst.reviews} == {inst.condition_name}

    def test_no_review_appears_twice_within_an_instance(self):
        for inst in build_instances(reviews_frame(), n_reviews=8, seed=0, max_instances=10):
            ids = [r["review_id"] for r in inst.reviews]
            assert len(ids) == len(set(ids))

    def test_drug_selection_is_not_the_alphabetical_prefix(self):
        # F4 #1, the regression that matters most.
        instances = build_instances(reviews_frame(), n_reviews=8, seed=0, max_instances=5)
        used = [inst.product_id for inst in instances]
        alphabetical_prefix = sorted({r for r in reviews_frame()["product_id"]})[: len(used)]
        assert used != alphabetical_prefix

    def test_is_deterministic_for_a_given_seed(self):
        frame = reviews_frame()
        a = build_instances(frame, n_reviews=8, seed=11, max_instances=10)
        b = build_instances(frame, n_reviews=8, seed=11, max_instances=10)
        assert [i.review_ids for i in a] == [i.review_ids for i in b]

    def test_skips_groups_with_too_few_reviews(self):
        frame = reviews_frame(n_drugs=3, per_drug=4)
        assert build_instances(frame, n_reviews=8, seed=0) == []

    def test_restricting_to_a_drug_set_excludes_all_others(self):
        frame = reviews_frame()
        instances = build_instances(frame, n_reviews=8, seed=0, drugs={"aaa", "bbb"})
        assert {i.product_id for i in instances} <= {"aaa", "bbb"}

    def test_instance_ids_are_unique(self):
        instances = build_instances(reviews_frame(), n_reviews=8, seed=0, max_instances=20)
        ids = [i.instance_id for i in instances]
        assert len(ids) == len(set(ids))


class TestAdverseEventAggregation:
    def test_collects_the_union_of_events_across_reviews(self):
        per_review = [{"dry skin"}, {"dry skin", "nausea"}, set()]
        assert instance_adverse_events(per_review) == {"dry skin", "nausea"}

    def test_counts_how_many_reviews_report_each_event(self):
        per_review = [{"dry skin"}, {"dry skin", "nausea"}, set()]
        _, counts = instance_adverse_events(per_review, with_counts=True)
        assert counts == {"dry skin": 2, "nausea": 1}


class TestAeConditionClassification:
    def test_three_or_more_singleton_events_makes_it_ae_singleton(self):
        per_review = [{"a"}, {"b"}, {"c"}, {"shared"}, {"shared"}]
        assert classify_ae_condition(per_review) == "ae_singleton"

    def test_many_shared_events_makes_it_ae_dense(self):
        per_review = [{"a", "b", "c", "d", "e"}, {"a", "b", "c", "d", "e"}]
        assert classify_ae_condition(per_review) == "ae_dense"

    def test_few_shared_events_makes_it_ae_sparse(self):
        per_review = [{"a"}, {"a"}, {"a"}]
        assert classify_ae_condition(per_review) == "ae_sparse"

    def test_no_events_is_sparse_not_an_error(self):
        assert classify_ae_condition([set(), set()]) == "ae_sparse"
