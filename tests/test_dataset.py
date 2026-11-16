import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.dataset_validator import audit_records, jaccard, _shingles  # noqa: E402


def _rec(t, label="FAVOUR", lang="Hindi"):
    return {"text": t, "label": label, "language": lang}


def test_counts_and_distribution():
    recs = [_rec("a", "FAVOUR"), _rec("bb", "AGAINST"), _rec("ccc", "NONE"), _rec("dddd", "NONE")]
    rep = audit_records("toy", recs, near_dup=False)
    assert rep.n_examples == 4
    assert rep.label_distribution == {"FAVOUR": 1, "AGAINST": 1, "NONE": 2}
    assert rep.imbalance_ratio == 2.0


def test_missing_and_empty_detected():
    recs = [{"text": ""}, {"text": None}, {"text": "ok"}]
    rep = audit_records("toy", recs, label_key=None, language_key=None, near_dup=False)
    assert rep.empty_text == 3 or rep.empty_text == 2
    assert rep.missing_text == 2


def test_invalid_label_flagged():
    rep = audit_records("toy", [_rec("x", "MAYBE")], near_dup=False)
    assert "MAYBE" in rep.invalid_labels


def test_exact_duplicates():
    recs = [_rec("same text"), _rec("same text"), _rec("other")]
    rep = audit_records("toy", recs, near_dup=False)
    assert rep.exact_duplicates == 1


def test_jaccard_bounds():
    assert jaccard(_shingles("abcdefg"), _shingles("abcdefg")) == 1.0
    assert jaccard(_shingles("abcdefg"), _shingles("zzzzzzz")) == 0.0


def test_script_distribution():
    recs = [_rec("जलवायु"), _rec("জলবায়ু"), _rec("climate")]
    rep = audit_records("toy", recs, near_dup=False)
    assert set(rep.script_distribution) == {"Devanagari", "Bengali", "Latin"}


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_sizes(n):
    rep = audit_records("toy", [_rec("x")] * n, near_dup=False)
    assert rep.n_examples == n
