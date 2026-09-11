"""Batched generation planning and record keeping.

v1 generated one prompt at a time: 540 short generations took 50 minutes on a 1B
model, which is why test-time scaling looked unaffordable. The k samples of one
instance share a prompt, so they batch together and share a prefill.

The model call itself needs a GPU; everything around it -- how work is grouped,
what gets recorded, how tokens are accounted -- is pure and tested here.
"""

import pytest

from fairsum.generation import (
    GenerationRecord,
    expand_requests,
    plan_batches,
    records_to_frame,
)


class TestPlanBatches:
    def test_splits_work_into_batches_of_the_requested_size(self):
        assert plan_batches(n_items=10, batch_size=4) == [(0, 4), (4, 8), (8, 10)]

    def test_a_single_batch_covers_everything_when_it_fits(self):
        assert plan_batches(n_items=3, batch_size=8) == [(0, 3)]

    def test_no_batches_for_no_work(self):
        assert plan_batches(n_items=0, batch_size=8) == []

    def test_rejects_a_nonpositive_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            plan_batches(n_items=4, batch_size=0)


class TestExpandRequests:
    def test_repeats_each_prompt_k_times(self):
        requests = expand_requests(["p1", "p2"], k=3)
        assert len(requests) == 6
        assert [r.prompt for r in requests].count("p1") == 3

    def test_tags_each_request_with_its_sample_index(self):
        requests = expand_requests(["p1"], k=3)
        assert sorted(r.sample_idx for r in requests) == [0, 1, 2]

    def test_tags_each_request_with_its_source_prompt_index(self):
        requests = expand_requests(["p1", "p2"], k=2)
        assert sorted({r.prompt_idx for r in requests}) == [0, 1]

    def test_groups_samples_of_one_prompt_together_for_prefill_reuse(self):
        # Contiguous runs per prompt keep the shared prefix inside one batch.
        indices = [r.prompt_idx for r in expand_requests(["p1", "p2"], k=3)]
        assert indices == [0, 0, 0, 1, 1, 1]

    def test_k_of_one_is_a_plain_pass_through(self):
        assert len(expand_requests(["p1", "p2"], k=1)) == 2

    def test_rejects_a_nonpositive_k(self):
        with pytest.raises(ValueError, match="k must be"):
            expand_requests(["p1"], k=0)


class TestGenerationRecord:
    def test_total_tokens_is_prompt_plus_completion(self):
        record = GenerationRecord(
            instance_id="i1",
            method="direct",
            sample_idx=0,
            raw="text",
            summary="text",
            parse_status="ok",
            prompt_tokens=100,
            completion_tokens=60,
            seed=0,
            model="m",
        )
        assert record.total_tokens == 160

    def test_records_carry_the_parse_status_so_failures_stay_visible(self):
        record = GenerationRecord(
            instance_id="i1",
            method="direct",
            sample_idx=0,
            raw="no marker here",
            summary=None,
            parse_status="no_marker",
            prompt_tokens=10,
            completion_tokens=5,
            seed=0,
            model="m",
        )
        assert record.summary is None
        assert record.parse_status == "no_marker"


class TestRecordsToFrame:
    def _records(self):
        return [
            GenerationRecord("i1", "direct", 0, "r", "s", "ok", 10, 5, 0, "m"),
            GenerationRecord("i1", "direct", 1, "r", None, "no_marker", 10, 7, 0, "m"),
        ]

    def test_one_row_per_record(self):
        assert len(records_to_frame(self._records())) == 2

    def test_completion_tokens_survive_for_compute_accounting(self):
        # H2 compares regimes per FLOP, so rejected samples must still be counted.
        assert records_to_frame(self._records())["completion_tokens"].sum() == 12

    def test_empty_input_gives_an_empty_frame_with_the_right_columns(self):
        frame = records_to_frame([])
        assert len(frame) == 0
        assert "parse_status" in frame.columns
