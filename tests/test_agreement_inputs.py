"""What reaches the agreement statistic.

Two ways the Day 4 gate could be wrong before any statistic is computed:

  * A review annotated by only one person drops out of the comparison without
    a trace. The report now names those reviews.
  * Labels are compared as exact strings, so "drowsy" against "drowsiness" is
    scored as a disagreement. Synonyms are merged through an explicit table.
    Merging can only remove disagreements, never add them, which is why the
    script reports agreement both before and after the table is applied.
"""

import pytest

from fairsum.annotation import agreement_report, canonicalise, load_synonym_table


class TestUnmatchedReviews:
    def test_reviews_found_in_only_one_file_are_listed(self):
        a = {"r1": {"x"}, "r2": {"y"}}
        b = {"r1": {"x"}, "r3": {"z"}}
        report = agreement_report(a, b)
        assert report.only_in_a == ["r2"]
        assert report.only_in_b == ["r3"]

    def test_the_summary_names_unmatched_reviews(self):
        report = agreement_report({"r1": {"x"}, "r2": {"y"}}, {"r1": {"x"}})
        assert "r2" in report.summary()

    def test_nothing_is_flagged_when_both_files_cover_the_same_reviews(self):
        report = agreement_report({"r1": {"x"}}, {"r1": {"x"}})
        assert report.only_in_a == [] and report.only_in_b == []
        assert "only one annotator" not in report.summary()


class TestCanonicalise:
    def test_merges_synonyms_through_the_table(self):
        merged = canonicalise({"r1": {"drowsy", "Dry Skin"}}, {"drowsy": "drowsiness"})
        assert merged == {"r1": {"drowsiness", "dry skin"}}

    def test_a_synonym_table_turns_a_label_mismatch_into_agreement(self):
        a, b = {"r1": {"drowsy"}}, {"r1": {"drowsiness"}}
        table = {"drowsy": "drowsiness"}
        assert agreement_report(a, b).n_disagreements == 2
        assert agreement_report(canonicalise(a, table), canonicalise(b, table)).n_disagreements == 0

    def test_merging_never_adds_disagreements(self):
        a = {"r1": {"drowsy", "nausea"}, "r2": {"rash"}, "r3": set()}
        b = {"r1": {"drowsiness"}, "r2": {"hives", "rash"}, "r3": {"sleepy"}}
        table = {"drowsy": "drowsiness", "sleepy": "drowsiness", "hives": "rash"}
        raw = agreement_report(a, b).n_disagreements
        merged = agreement_report(canonicalise(a, table), canonicalise(b, table)).n_disagreements
        assert merged <= raw

    def test_does_not_mutate_its_input(self):
        events = {"r1": {"drowsy"}}
        canonicalise(events, {"drowsy": "drowsiness"})
        assert events == {"r1": {"drowsy"}}


class TestLoadSynonymTable:
    def test_reads_surface_and_canonical_columns(self, tmp_path):
        path = tmp_path / "synonyms.csv"
        path.write_text(
            "surface,canonical\nDrowsy,Drowsiness\nfelt like a zombie,drowsiness\n",
            encoding="utf-8",
        )
        assert load_synonym_table(path) == {
            "drowsy": "drowsiness",
            "felt like a zombie": "drowsiness",
        }

    def test_rejects_a_surface_mapped_to_two_labels(self, tmp_path):
        path = tmp_path / "synonyms.csv"
        path.write_text("surface,canonical\ndrowsy,drowsiness\ndrowsy,fatigue\n", encoding="utf-8")
        with pytest.raises(ValueError, match="drowsy"):
            load_synonym_table(path)

    def test_requires_the_header(self, tmp_path):
        path = tmp_path / "synonyms.csv"
        path.write_text("a,b\nx,y\n", encoding="utf-8")
        with pytest.raises(ValueError, match="surface"):
            load_synonym_table(path)
