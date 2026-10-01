"""LaTeX helpers for the paper's generated numbers and tables.

Every number in the paper comes from these files, so a wrong sign, a lost
minus or an unescaped underscore here would silently corrupt the paper.
"""

import math

import pandas as pd
import pytest

from fairsum.kwsum.paper import fmt, fmt_ci, fmt_p, latex_table, macro_file, tex_escape


def test_tex_escape_handles_the_characters_condition_names_contain():
    assert tex_escape("kw_strat_neg") == r"kw\_strat\_neg"
    assert tex_escape("50% & more #1") == r"50\% \& more \#1"


@pytest.mark.parametrize("x, kw, expected", [
    (0.1578, {}, "0.158"),
    (-0.0734, {}, "$-$0.073"),
    (0.0402, {"signed": True}, "+0.040"),
    (-0.0, {"signed": True}, "0.000"),          # no "+0.000" or "-0.000"
    (104.33, {"digits": 1}, "104.3"),
    (0.625, {"percent": True, "digits": 1}, "62.5\\%"),
    (float("nan"), {}, "--"),
    (None, {}, "--"),
])
def test_fmt(x, kw, expected):
    assert fmt(x, **kw) == expected


def test_fmt_ci_signs_the_estimate_and_keeps_the_interval_order():
    assert fmt_ci(0.0402, 0.0174, 0.0616) == "+0.040 [0.017, 0.062]"
    assert fmt_ci(-0.058, -0.082, -0.034) == "$-$0.058 [$-$0.082, $-$0.034]"


@pytest.mark.parametrize("p, expected", [(0.0003, "$<$0.001"), (0.0581, "0.058"), (1.0, "1.000"),
                                         (math.nan, "--")])
def test_fmt_p(p, expected):
    assert fmt_p(p) == expected


def test_macro_file_defines_each_key_and_rejects_unsafe_names():
    text = macro_file({"mean-unsup-l3b-direct": "0.761", "n-models": "5"})
    assert r"\expandafter\def\csname kw@mean-unsup-l3b-direct\endcsname{0.761}" in text
    assert r"\expandafter\def\csname kw@n-models\endcsname{5}" in text
    with pytest.raises(ValueError, match="key"):
        macro_file({"bad key!": "1"})


def test_latex_table_is_booktabs_with_escaped_cells():
    df = pd.DataFrame({"Prompt": ["kw_global", "direct"], "MA": ["$-$0.050", "$-$0.073"]})
    out = latex_table(df, caption="Majority amplification.", label="tab:ma", align="lr")
    assert r"\begin{tabular}{lr}" in out and r"\toprule" in out and r"\bottomrule" in out
    assert r"kw\_global & $-$0.050 \\" in out          # cell text escaped, math left alone
    assert r"\caption{Majority amplification.}" in out and r"\label{tab:ma}" in out


def test_latex_table_can_tighten_column_spacing_for_wide_tables():
    df = pd.DataFrame({"a": ["1"], "b": ["2"]})
    assert r"\setlength{\tabcolsep}{3pt}" in latex_table(df, "c", "tab:x", colsep="3pt")
    assert r"\tabcolsep" not in latex_table(df, "c", "tab:x")
