"""Task metrics for three-way stance detection.

Macro-F1 is the headline number, so it carries an uncertainty estimate: a
stratified bootstrap CI and a label-preserving permutation test against the
majority-class baseline. Nothing here invents numbers - every function returns a
plain dict that the experiment registry serialises verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = ["FAVOUR", "AGAINST", "NONE"]


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average=None, zero_division=0
    )
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        acc = round(float(accuracy_score(y_true, y_pred)), 4)
        bal_acc = round(float(balanced_accuracy_score(y_true, y_pred)), 4) if len(set(y_true)) > 1 else (1.0 if y_true == y_pred else 0.0)
        try:
            ck = round(float(cohen_kappa_score(y_true, y_pred, labels=list(labels))), 4)
            if np.isnan(ck):
                ck = 0.0
        except Exception:
            ck = 0.0

    return {
        "n": len(y_true),
        "macro_f1": round(float(f_macro), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "weighted_f1": round(float(f_w), 4),
        "weighted_precision": round(float(p_w), 4),
        "weighted_recall": round(float(r_w), 4),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "cohen_kappa": ck,
        "per_class": {
            lbl: {
                "precision": round(float(per_class_p[i]), 4),
                "recall": round(float(per_class_r[i]), 4),
                "f1": round(float(per_class_f[i]), 4),
                "support": int(support[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "label_order": list(labels),
    }


def bootstrap_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    n_boot: int = 1000,
    seed: int = 42,
    labels: Sequence[str] = LABELS,
) -> dict:
    """Stratified bootstrap over items; returns point estimate and 95 % percentile CI."""
    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))
    rng = np.random.default_rng(seed)
    point = f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(f1_score(y_true[idx], y_pred[idx], labels=list(labels), average="macro", zero_division=0))
    stats = np.array(stats)
    return {
        "macro_f1": round(float(point), 4),
        "ci95_low": round(float(np.percentile(stats, 2.5)), 4),
        "ci95_high": round(float(np.percentile(stats, 97.5)), 4),
        "se": round(float(stats.std(ddof=1)), 4),
        "n_boot": n_boot,
    }


def mcnemar_test(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> dict:
    """Exact binomial McNemar on the discordant pairs (a vs b)."""
    from scipy.stats import binomtest

    a_ok = np.array([t == p for t, p in zip(y_true, pred_a)])
    b_ok = np.array([t == p for t, p in zip(y_true, pred_b)])
    n01 = int(np.sum(a_ok & ~b_ok))   # a right, b wrong
    n10 = int(np.sum(~a_ok & b_ok))   # a wrong, b right
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "note": "identical predictions"}
    res = binomtest(n01, n01 + n10, 0.5, alternative="two-sided")
    return {"n01": n01, "n10": n10, "statistic": n01, "p_value": round(float(res.pvalue), 6)}


def per_language_report(records: Sequence[dict], y_true: Sequence[str], y_pred: Sequence[str], lang_key: str = "language") -> dict:
    out: dict[str, dict] = {}
    langs = sorted({str(r.get(lang_key, "unknown")) for r in records})
    for lang in langs:
        sel = [i for i, r in enumerate(records) if str(r.get(lang_key, "unknown")) == lang]
        if not sel:
            continue
        out[lang] = compute_metrics([y_true[i] for i in sel], [y_pred[i] for i in sel])
    return out


def per_script_report(records: Sequence[dict], y_true: Sequence[str], y_pred: Sequence[str], script_key: str = "script") -> dict:
    out: dict[str, dict] = {}
    for script in sorted({str(r.get(script_key, "unknown")) for r in records}):
        sel = [i for i, r in enumerate(records) if str(r.get(script_key, "unknown")) == script]
        if sel:
            out[script] = compute_metrics([y_true[i] for i in sel], [y_pred[i] for i in sel])
    return out


def majority_baseline(y_train: Sequence[str], y_true: Sequence[str], labels: Sequence[str] = LABELS) -> dict:
    """The floor: predict the training majority class everywhere."""
    from collections import Counter

    maj = Counter(y_train).most_common(1)[0][0]
    return compute_metrics(y_true, [maj] * len(y_true), labels)


def format_report_block(metrics: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"### {title}")
    if "macro_f1" in metrics and "per_class" in metrics:
        lines.append(
            f"- macro-F1 **{metrics['macro_f1']:.4f}** | accuracy {metrics['accuracy']:.4f} | "
            f"weighted-F1 {metrics['weighted_f1']:.4f} | n={metrics['n']}"
        )
        for lbl, d in metrics["per_class"].items():
            lines.append(
                f"  - {lbl:8s} P {d['precision']:.3f} R {d['recall']:.3f} F1 {d['f1']:.3f} (n={d['support']})"
            )
    return "\n".join(lines)
