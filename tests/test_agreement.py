import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.agreement import cohens_kappa, krippendorff_alpha_nominal  # noqa: E402


def test_perfect_agreement():
    a = ["FAVOUR", "AGAINST", "NONE", "FAVOUR"]
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    assert krippendorff_alpha_nominal([[x, x] for x in a]) == pytest.approx(1.0)


def test_total_disagreement_is_negative():
    a = ["FAVOUR", "AGAINST"]
    b = ["AGAINST", "FAVOUR"]
    assert cohens_kappa(a, b) == pytest.approx(-1.0)


def test_known_kappa_value():
    # 2x2 table: marginals are 25/25 (A) and 30/20 (B); 35 of 50 pairs agree.
    # po = 0.70, pe = 0.5*0.6 + 0.5*0.4 = 0.50 -> kappa = (0.70-0.50)/(1-0.50) = 0.40
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.40, abs=1e-9)


def test_kappa_chance_corrected_not_raw_agreement():
    # a: 40 y / 10 n ; b: 30 y / 20 n ; po = 40/50 = 0.80
    # pe = 0.8*0.6 + 0.2*0.4 = 0.56  ->  kappa = 0.24/0.44 = 0.5454545...
    a = ["y"] * 40 + ["n"] * 10
    b = ["y"] * 30 + ["n"] * 10 + ["n"] * 10
    assert cohens_kappa(a, b) == pytest.approx(0.24 / 0.44, abs=1e-9)


def test_alpha_handles_missing():
    units = [["FAVOUR", "FAVOUR"], ["AGAINST"], ["NONE", "NONE"], ["FAVOUR", "NONE"]]
    a = krippendorff_alpha_nominal(units)
    assert -1.0 <= a <= 1.0


def test_empty_units_nan():
    import math

    assert math.isnan(krippendorff_alpha_nominal([["FAVOUR"], ["NONE"]]))
