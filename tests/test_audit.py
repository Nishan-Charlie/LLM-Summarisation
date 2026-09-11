"""The Day 1 diagnostic: is v1 measuring fairness or judge parse failures?

`implied_degenerate_fraction` is the estimator derived in the vault note
`F1 Degenerate Judge Rows`. It backs the degenerate rate out of published
aggregates alone, which is how the hypothesis was formed before any code ran.
`degenerate_fraction` measures the same quantity directly from the rows, and
Day 1 is the comparison of the two.
"""

import pandas as pd
import pytest

from fairsum.audit import (
    audit_results,
    degenerate_fraction,
    identity_residuals,
    implied_degenerate_fraction,
    parse_counts,
)


class TestParseCounts:
    def test_parses_a_json_object(self):
        assert parse_counts('{"a": 1, "b": 2}') == {"a": 1, "b": 2}

    def test_parses_a_python_repr_dict(self):
        # v1 wrote some columns with str(dict), giving single quotes.
        assert parse_counts("{'Highly Effective': 3}") == {"Highly Effective": 3}

    def test_passes_a_dict_through(self):
        assert parse_counts({"a": 1}) == {"a": 1}

    def test_returns_empty_for_unparseable_text(self):
        assert parse_counts("not a dict at all") == {}


class TestDegenerateFraction:
    def test_counts_rows_whose_counts_carry_no_mass(self):
        rows = [{"a": 0, "b": 0}, {"a": 1, "b": 0}, {"a": 0, "b": 0}]
        assert degenerate_fraction(rows) == pytest.approx(2 / 3)

    def test_is_zero_when_every_row_has_mass(self):
        assert degenerate_fraction([{"a": 1}, {"b": 2}]) == pytest.approx(0.0)

    def test_treats_an_empty_dict_as_degenerate(self):
        assert degenerate_fraction([{}]) == pytest.approx(1.0)


class TestImpliedDegenerateFraction:
    def test_recovers_the_v1_direct_method_estimate(self):
        # method_table row for `direct` in drug_summary_01_Method_A.ipynb.
        f = implied_degenerate_fraction(mean_spd=0.1761, mean_uer=0.1358, n_labels=5)
        assert f == pytest.approx(0.4775, abs=1e-3)

    def test_is_zero_when_the_identity_holds_exactly(self):
        # spd == 2 * uer means no degenerate rows in the mixture.
        f = implied_degenerate_fraction(mean_spd=0.20, mean_uer=0.10, n_labels=5)
        assert f == pytest.approx(0.0, abs=1e-9)

    def test_is_one_when_every_row_is_degenerate(self):
        # A degenerate row scores spd == uer == 1/r.
        f = implied_degenerate_fraction(mean_spd=0.2, mean_uer=0.2, n_labels=5)
        assert f == pytest.approx(1.0)

    def test_recovers_a_known_mixture(self):
        # Build a mixture by hand: 30% degenerate, 70% valid with uer = 0.08.
        f_true, u = 0.30, 0.08
        mean_uer = f_true * 0.2 + (1 - f_true) * u
        mean_spd = f_true * 0.2 + (1 - f_true) * 2 * u
        assert implied_degenerate_fraction(mean_spd, mean_uer, 5) == pytest.approx(f_true)


class TestIdentityResiduals:
    def test_residual_is_zero_for_rows_obeying_spd_equals_two_uer(self):
        df = pd.DataFrame({"SPD": [0.2, 0.1], "UER": [0.1, 0.05]})
        assert identity_residuals(df).abs().max() == pytest.approx(0.0)

    def test_residual_is_nonzero_for_a_degenerate_row(self):
        df = pd.DataFrame({"SPD": [0.2], "UER": [0.2]})
        assert identity_residuals(df).iloc[0] == pytest.approx(-0.2)


class TestAuditResults:
    def _frame(self):
        return pd.DataFrame(
            {
                "method": ["direct", "direct", "ssfp", "ssfp"],
                "summary_counts": [
                    {"Highly Effective": 0, "Ineffective": 0},
                    {"Highly Effective": 4, "Ineffective": 4},
                    {"Highly Effective": 2, "Ineffective": 6},
                    {"Highly Effective": 4, "Ineffective": 4},
                ],
                "SPD": [0.2, 0.0, 0.05, 0.0],
                "UER": [0.2, 0.0, 0.025, 0.0],
            }
        )

    def test_reports_the_overall_degenerate_rate(self):
        report = audit_results(self._frame())
        assert report.overall_degenerate_rate == pytest.approx(0.25)

    def test_reports_the_rate_per_method(self):
        report = audit_results(self._frame())
        assert report.by_method.loc["direct", "degenerate_rate"] == pytest.approx(0.5)
        assert report.by_method.loc["ssfp", "degenerate_rate"] == pytest.approx(0.0)

    def test_flags_that_the_rate_differs_across_methods(self):
        assert audit_results(self._frame()).rate_varies_by_method is True

    def test_does_not_flag_variation_when_rates_match(self):
        df = self._frame()
        df.at[2, "summary_counts"] = {"Highly Effective": 0, "Ineffective": 0}
        assert audit_results(df).rate_varies_by_method is False

    def test_verdict_is_void_when_the_degenerate_rate_is_material(self):
        assert audit_results(self._frame()).verdict == "void"

    def test_verdict_is_clean_when_no_rows_are_degenerate(self):
        df = self._frame().iloc[[1, 2, 3]].reset_index(drop=True)
        assert audit_results(df).verdict == "clean"
