"""Grouping mode, and the indication claim in the prompt.

Discovered by running `build_pool.py` on the real pool: Druglib's `condition` is
free text averaging 4.5 distinct indications per drug (max 30), so grouping on
(drug, indication) leaves only 42 groups with >= 8 reviews, against 144 when
grouping by drug alone. The F4 #5 fix as originally specified would have cost
70% of the available instances.

Resolution: group by drug, and do not name an indication in the prompt. v1's bug
was *asserting* an indication the reviews were not about; a prompt that makes no
such claim cannot make a false one, and the reviews carry their own context.
"""

import pandas as pd
import pytest

from fairsum.instances import Instance, build_instances
from fairsum.prompts import PROMPT_BUILDERS


def mixed_indication_frame(n_drugs=4, per_indication=5, indications=3):
    rows = []
    for i in range(n_drugs):
        drug = chr(ord("a") + i) * 3
        for c in range(indications):
            for j in range(per_indication):
                rows.append(
                    {
                        "review_id": f"{drug}-{c}-{j}",
                        "product_id": drug,
                        "condition_name": f"cond{c}",
                        "text": f"review {j}",
                    }
                )
    return pd.DataFrame(rows)


class TestGroupByDrug:
    def test_drug_grouping_finds_instances_that_indication_grouping_cannot(self):
        # 5 reviews per indication, so no indication reaches 8 on its own.
        frame = mixed_indication_frame()
        by_indication = build_instances(frame, n_reviews=8, seed=0, group_by="drug_indication")
        by_drug = build_instances(frame, n_reviews=8, seed=0, group_by="drug")
        assert by_indication == []
        assert len(by_drug) > 0

    def test_drug_grouping_still_never_mixes_drugs(self):
        for inst in build_instances(
            mixed_indication_frame(), n_reviews=8, seed=0, group_by="drug"
        ):
            assert {r["product_id"] for r in inst.reviews} == {inst.product_id}

    def test_drug_grouping_leaves_the_indication_unset_rather_than_guessing(self):
        # v1 took the first review's condition and asserted it for all of them.
        for inst in build_instances(
            mixed_indication_frame(), n_reviews=8, seed=0, group_by="drug"
        ):
            assert inst.condition_name is None

    def test_drug_grouping_records_which_indications_are_present(self):
        instances = build_instances(
            mixed_indication_frame(), n_reviews=8, seed=0, group_by="drug"
        )
        assert instances[0].metadata["indications"]

    def test_indication_grouping_still_pins_a_single_indication(self):
        frame = mixed_indication_frame(per_indication=10)
        for inst in build_instances(
            frame, n_reviews=8, seed=0, group_by="drug_indication"
        ):
            assert inst.condition_name is not None
            assert {r["condition_name"] for r in inst.reviews} == {inst.condition_name}

    def test_rejects_an_unknown_grouping_mode(self):
        with pytest.raises(ValueError, match="group_by"):
            build_instances(mixed_indication_frame(), group_by="something_else")


class TestPromptsWithoutAnIndication:
    def _instance(self, condition_name):
        return Instance(
            instance_id="i1",
            product_id="lyrica",
            condition_name=condition_name,
            reviews=[{"review_id": f"r{i}", "text": f"body {i}"} for i in range(8)],
        )

    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_prompt_makes_no_indication_claim_when_none_is_known(self, name):
        prompt = PROMPT_BUILDERS[name](self._instance(None))
        assert "used to treat" not in prompt
        assert "None" not in prompt

    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_prompt_still_names_the_indication_when_one_is_known(self, name):
        prompt = PROMPT_BUILDERS[name](self._instance("fibromyalgia"))
        assert "fibromyalgia" in prompt

    @pytest.mark.parametrize("name", sorted(PROMPT_BUILDERS))
    def test_prompt_is_still_well_formed_without_an_indication(self, name):
        prompt = PROMPT_BUILDERS[name](self._instance(None))
        assert "8 patient reviews" in prompt
        assert "60 words" in prompt
