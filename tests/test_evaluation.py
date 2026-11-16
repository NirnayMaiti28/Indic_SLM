import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    compute_metrics,
    per_language_report,
    per_script_report,
    bootstrap_macro_f1,
    majority_baseline,
)


def test_compute_metrics_perfect():
    y_true = ["FAVOUR", "AGAINST", "NONE", "FAVOUR"]
    y_pred = ["FAVOUR", "AGAINST", "NONE", "FAVOUR"]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["per_class"]["FAVOUR"]["f1"] == 1.0


def test_per_language_metrics():
    records = [
        {"language": "Hindi", "script": "Devanagari"},
        {"language": "Hindi", "script": "Devanagari"},
        {"language": "Bengali", "script": "Bengali"},
        {"language": "Bengali", "script": "Bengali"},
    ]
    y_true = ["FAVOUR", "AGAINST", "FAVOUR", "NONE"]
    y_pred = ["FAVOUR", "AGAINST", "FAVOUR", "AGAINST"]

    rep = per_language_report(records, y_true, y_pred)
    assert "Hindi" in rep
    assert "Bengali" in rep
    assert rep["Hindi"]["accuracy"] == 1.0
    assert rep["Bengali"]["accuracy"] == 0.5


def test_bootstrap_ci():
    y_true = ["FAVOUR", "AGAINST", "NONE"] * 10
    y_pred = ["FAVOUR", "AGAINST", "NONE"] * 10
    boot = bootstrap_macro_f1(y_true, y_pred, n_boot=50, seed=42)
    assert "macro_f1" in boot
    assert "ci95_low" in boot
    assert "ci95_high" in boot
    assert boot["macro_f1"] == 1.0
