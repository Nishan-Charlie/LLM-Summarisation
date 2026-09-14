"""Exporting an annotation batch to a spreadsheet and reading it back.

Needs openpyxl (the `annotate` extra) and is skipped without it. The parsing
rules are tested in `test_annotation_sheet.py`; this file checks only that the
sheet carries them.
"""

import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")

from fairsum.annotation import AnnotationSheetError, Certainty  # noqa: E402
from fairsum.annotation_sheet import export_sheet, load_sheet_annotations  # noqa: E402


def batch():
    return pd.DataFrame(
        [
            {
                "review_id": "r1",
                "product_id": "biaxin",
                "condition_name": "sinusitis",
                "text": "Worked, but gave me terrible nausea.",
            },
            {
                "review_id": "r2",
                "product_id": "voltaren",
                "condition_name": "arthritis",
                "text": "No side effects at all.",
            },
        ]
    )


def fill(path, values):
    """Write {review_id: {column: value}} into an exported sheet, as an annotator would."""
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    header = [cell.value for cell in sheet[1]]
    for cells in sheet.iter_rows(min_row=2):
        review_id = cells[header.index("review_id")].value
        for column, value in values.get(review_id, {}).items():
            cells[header.index(column)].value = value
    workbook.save(path)


class TestExportSheet:
    def test_has_one_row_per_review_carrying_the_text(self, tmp_path):
        path = tmp_path / "sheet.xlsx"
        export_sheet(batch(), path)
        sheet = openpyxl.load_workbook(path).active
        header = [cell.value for cell in sheet[1]]
        rows = list(sheet.iter_rows(min_row=2, values_only=True))
        assert len(rows) == 2
        assert rows[0][header.index("text")] == "Worked, but gave me terrible nausea."

    def test_certainty_and_attribution_cells_offer_dropdowns(self, tmp_path):
        path = tmp_path / "sheet.xlsx"
        export_sheet(batch(), path)
        sheet = openpyxl.load_workbook(path).active
        formulas = [v.formula1 for v in sheet.data_validations.dataValidation]
        assert any("negated" in f for f in formulas)
        assert any("condition" in f for f in formulas)

    def test_refuses_to_overwrite_an_existing_sheet(self, tmp_path):
        # Re-running export over a half-annotated sheet would destroy the labels.
        path = tmp_path / "sheet.xlsx"
        export_sheet(batch(), path)
        with pytest.raises(FileExistsError):
            export_sheet(batch(), path)


class TestLoadSheetAnnotations:
    def test_a_filled_sheet_round_trips_to_annotations(self, tmp_path):
        path = tmp_path / "sheet.xlsx"
        export_sheet(batch(), path)
        fill(
            path,
            {
                "r1": {
                    "done": "yes",
                    "span_1": "terrible nausea",
                    "event_1": "Nausea",
                    "certainty_1": "asserted",
                    "attribution_1": "drug",
                },
                "r2": {
                    "done": "yes",
                    "span_1": "No side effects at all",
                    "event_1": "side effects",
                    "certainty_1": "negated",
                    "attribution_1": "drug",
                },
            },
        )
        by_id = {a.review_id: a for a in load_sheet_annotations(path)}
        assert by_id["r1"].event_set == {"nausea"}
        assert by_id["r2"].events[0].certainty is Certainty.NEGATED
        assert by_id["r2"].event_set == set()

    def test_an_untouched_sheet_is_rejected_not_read_as_no_events(self, tmp_path):
        path = tmp_path / "sheet.xlsx"
        export_sheet(batch(), path)
        with pytest.raises(AnnotationSheetError) as exc:
            load_sheet_annotations(path)
        assert len(exc.value.problems) == 2
