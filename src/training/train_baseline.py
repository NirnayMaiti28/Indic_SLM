"""Traditional ML Baseline Training & Benchmarking (Phase 7).

Trains and evaluates:
- Majority baseline (floor)
- TF-IDF (word n-grams) + Logistic Regression & SVM
- TF-IDF (char n-grams) + Logistic Regression & SVM
- TF-IDF (combined word+char) + Logistic Regression & SVM

Supports:
  python -m src.training.train_baseline --dataset indic
  python -m src.training.train_baseline --dataset gwstance
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_loader import load_custom, load_gwstance
from src.data.preprocessing import preprocess
from src.data.splitting import group_stratified_split
from src.evaluation.metrics import compute_metrics, per_language_report
from src.models.baseline import TfidfConfig, TfidfStanceClassifier, majority_predictor
from src.utils.registry import log_experiment

class MajorityClassifier:
    def __init__(self):
        self.majority_label = "NONE"

    def fit(self, X, y):
        from collections import Counter
        self.majority_label = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X):
        return [self.majority_label] * len(X)


SPLITS_PATH = ROOT / "data" / "splits" / "indic_climate_splits.json"


def get_data(dataset_name: str = "indic", seed: int = 42):
    if dataset_name == "indic":
        if SPLITS_PATH.exists():
            with open(SPLITS_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return data["splits"]
        recs = load_custom()
        if not recs:
            raise FileNotFoundError("Indic dataset not found. Run: python -m src.data.generate_indic_dataset")
        for r in recs:
            r["text_raw"] = r["text"]
            r["text"] = preprocess(r["text"], variant="minimal").text
        splits, _ = group_stratified_split(recs, fractions=(0.70, 0.15, 0.15), seed=seed)
        return splits
    elif dataset_name == "gwstance":
        recs = load_gwstance(drop_screening=True)
        for r in recs:
            r["text_raw"] = r["text"]
            r["text"] = preprocess(r["text"], variant="minimal").text
        splits, _ = group_stratified_split(recs, fractions=(0.70, 0.15, 0.15), seed=seed)
        return splits
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_experiment(
    clf,
    name: str,
    train_recs: list[dict],
    val_recs: list[dict],
    test_recs: list[dict],
    dataset_name: str,
    notes: str = "",
) -> dict:
    X_tr = [r["text"] for r in train_recs]
    y_tr = [r["label"] for r in train_recs]
    X_va = [r["text"] for r in val_recs]
    y_va = [r["label"] for r in val_recs]
    X_te = [r["text"] for r in test_recs]
    y_te = [r["label"] for r in test_recs]

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    train_sec = round(time.time() - t0, 3)

    t0 = time.time()
    preds_te = clf.predict(X_te)
    infer_ms = round((time.time() - t0) * 1000 / max(1, len(X_te)), 3)

    preds_va = clf.predict(X_va)
    m_va = compute_metrics(y_va, preds_va)
    m_te = compute_metrics(y_te, preds_te)

    per_lang = per_language_report(test_recs, y_te, preds_te)
    hi_f1 = per_lang.get("Hindi", {}).get("macro_f1")
    bn_f1 = per_lang.get("Bengali", {}).get("macro_f1")
    en_f1 = per_lang.get("English", {}).get("macro_f1")

    log_experiment(
        model=name,
        dataset=dataset_name,
        split_strategy="group_stratified",
        n_train=len(train_recs),
        n_val=len(val_recs),
        n_test=len(test_recs),
        preprocessing="minimal",
        macro_f1=m_te["macro_f1"],
        accuracy=m_te["accuracy"],
        weighted_f1=m_te["weighted_f1"],
        f1_favour=m_te.get("per_class", {}).get("FAVOUR", {}).get("f1", 0.0),
        f1_against=m_te.get("per_class", {}).get("AGAINST", {}).get("f1", 0.0),
        f1_none=m_te.get("per_class", {}).get("NONE", {}).get("f1", 0.0),
        hindi_f1=hi_f1,
        bengali_f1=bn_f1,
        english_f1=en_f1,
        train_seconds=train_sec,
        infer_ms_per_item=infer_ms,
        notes=f"val_macro_f1={m_va['macro_f1']:.4f}; {notes}".strip("; "),
    )

    return {
        "model": name,
        "val_macro_f1": m_va["macro_f1"],
        "test_macro_f1": m_te["macro_f1"],
        "test_accuracy": m_te["accuracy"],
        "hindi_f1": hi_f1,
        "bengali_f1": bn_f1,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train classical ML baselines")
    ap.add_argument("--dataset", choices=["indic", "gwstance"], default="indic",
                    help="Dataset to evaluate baselines on (default: indic)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print(f"[*] Loading dataset: {args.dataset}...")
    splits = get_data(args.dataset, seed=args.seed)
    train_recs = splits["train"]
    val_recs = splits["val"]
    test_recs = splits["test"]

    print(f"[+] Loaded: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test.")

    configs = [
        ("majority_class", MajorityClassifier(), "floor baseline"),
        ("tfidf_word_logreg", TfidfStanceClassifier(TfidfConfig(feature="word", model="logreg", C=1.0)), "word n-grams 1-2"),
        ("tfidf_word_svm", TfidfStanceClassifier(TfidfConfig(feature="word", model="svm", C=1.0)), "word n-grams 1-2, linear SVM"),
        ("tfidf_char_logreg", TfidfStanceClassifier(TfidfConfig(feature="char", model="logreg", C=1.0)), "char n-grams 2-5"),
        ("tfidf_char_svm", TfidfStanceClassifier(TfidfConfig(feature="char", model="svm", C=1.0)), "char n-grams 2-5, linear SVM"),
        ("tfidf_combined_logreg", TfidfStanceClassifier(TfidfConfig(feature="combined", model="logreg", C=1.0)), "word+char combined"),
        ("tfidf_combined_svm", TfidfStanceClassifier(TfidfConfig(feature="combined", model="svm", C=1.0)), "word+char combined, linear SVM"),
    ]

    print("\n" + "=" * 80)
    print(f"{'Model':<25} | {'Val Macro-F1':<12} | {'Test Macro-F1':<14} | {'Test Acc':<10} | {'Hindi F1':<10} | {'Bengali F1':<10}")
    print("=" * 80)

    for name, clf, notes in configs:
        res = run_experiment(clf, name, train_recs, val_recs, test_recs, args.dataset, notes)
        hi_str = f"{res['hindi_f1']:.4f}" if res['hindi_f1'] is not None else "N/A"
        bn_str = f"{res['bengali_f1']:.4f}" if res['bengali_f1'] is not None else "N/A"
        print(f"{name:<25} | {res['val_macro_f1']:<12.4f} | {res['test_macro_f1']:<14.4f} | {res['test_accuracy']:<10.4f} | {hi_str:<10} | {bn_str:<10}")

    print("=" * 80)
    print("[+] All baseline results logged to experiments/experiment_registry.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
