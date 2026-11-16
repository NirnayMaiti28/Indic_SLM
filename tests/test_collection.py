import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.collection import (  # noqa: E402
    CollectionPlan, coverage_report, dedup_rows, validate_rows,
)


def _row(**kw):
    base = {
        "id": "hi_001", "text": "जलवायु परिवर्तन गंभीर समस्या है", "language": "Hindi",
        "script": "Devanagari", "label": "FAVOUR", "annotation_status": "human_verified",
        "source": "youtube_public_comments",
    }
    base.update(kw)
    return base


def test_valid_row_accepted():
    v = validate_rows([_row()])
    assert v.n_ok == 1 and not v.errors


def test_missing_field_rejected():
    r = _row(); del r["label"]
    v = validate_rows([r])
    assert v.n_ok == 0
    assert v.errors[0]["error"] == "missing_fields"


def test_invalid_label_rejected():
    v = validate_rows([_row(label="MAYBE")])
    assert v.errors[0]["error"] == "invalid_label"


def test_synthetic_excluded_from_gold():
    v = validate_rows([_row(annotation_status="synthetic")])
    assert v.errors[0]["error"] == "synthetic_in_gold_split"
    v2 = validate_rows([_row(annotation_status="synthetic")], require_human=False)
    assert v2.n_ok == 1


def test_pending_status_rejected():
    v = validate_rows([_row(annotation_status="pending_human_annotation")])
    assert v.errors[0]["error"] == "not_human_verified"


def test_duplicate_id_rejected():
    v = validate_rows([_row(), _row()])
    assert v.n_ok == 1 and v.errors[0]["error"] == "duplicate_id"


def test_handle_and_url_rejected():
    assert validate_rows([_row(text="dekh @rahul kya bola")]).errors[0]["error"] == "handle_present"
    assert validate_rows([_row(text="see https://x.co/a")]).errors[0]["error"] == "raw_url_present"


def test_label_normalised_to_upper():
    v = validate_rows([_row(label="favour")])
    assert v.ok[0]["label"] == "FAVOUR"


def test_dedup_exact_and_near():
    rows = [_row(id="a"), _row(id="b"), _row(id="c", text="something entirely different about football")]
    kept, stats = dedup_rows(rows)
    assert stats["exact_duplicates_removed"] == 1
    assert stats["kept"] == 2


def test_coverage_reports_gaps():
    rep = coverage_report([_row()])
    assert rep["n_rows"] == 1
    assert rep["complete"] is False
    assert rep["remaining_total"] > 0
    hindi = [s for s in rep["quota_slots"] if s["language"] == "Hindi" and s["slice"] == "Devanagari"][0]
    assert hindi["have"] == 1 and hindi["remaining"] == 1199


def test_class_balance_floor_flag():
    rep = coverage_report([_row(), _row(id="b", label="NONE")])
    assert rep["class_balance"]["Hindi"]["AGAINST"]["below_floor"] is True


def test_empty_corpus():
    rep = coverage_report([])
    assert rep["n_rows"] == 0 and rep["complete"] is False


def test_plan_targets_shape():
    t = CollectionPlan().targets()
    assert set(t) == {"Hindi", "Bengali"}
    assert t["Hindi"]["Devanagari"] == 1200
