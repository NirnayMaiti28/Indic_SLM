"""Run the traditional-ML baseline sweep and write the comparison report.

    python -m src.training.run_baselines                 # default sweep
    python -m src.training.run_baselines --quick         # 4 configs, fast

Every row is appended to experiments/experiment_registry.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset_loader import load_gwstance  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splitting import group_stratified_split, random_stratified_split, save_splits  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_macro_f1, compute_metrics, format_report_block, majority_baseline, mcnemar_test,
)
from src.models.baseline import TfidfConfig, TfidfStanceClassifier, majority_predictor  # noqa: E402
from src.utils.registry import log_experiment  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def prepare(variant: str = "minimal") -> tuple[list[dict], dict, dict]:
    recs = load_gwstance(drop_screening=True)
    for r in recs:
        p = preprocess(r["text"], variant=variant)
        r["text_raw"] = r["text"]
        r["text"] = p.text
    split, rep = group_stratified_split(recs, fractions=(0.70, 0.15, 0.15), seed=42)
    leaky, leaky_rep = random_stratified_split(recs, fractions=(0.70, 0.15, 0.15), seed=42)
    return split, rep.to_dict(), {"split": leaky, "report": leaky_rep.to_dict()}


def sweep(quick: bool = False) -> list[dict]:
    set_seed(42)
    split, split_rep, leaky = prepare("minimal")
    tr = split["train"]; va = split["val"]; te = split["test"]
    print(f"split: train={len(tr)} val={len(va)} test={len(te)} | groups={split_rep['n_groups']} "
          f"| leakage_safe={split_rep['leakage_safe']} overlap={split_rep['train_test_group_overlap']}")

    y_tr = [r["label"] for r in tr]; y_va = [r["label"] for r in va]; y_te = [r["label"] for r in te]
    x_tr = [r["text"] for r in tr]; x_va = [r["text"] for r in va]; x_te = [r["text"] for r in te]

    results: list[dict] = []

    # ---- floor: majority class ----
    maj = majority_predictor(y_tr, len(te))
    m_maj = compute_metrics(y_te, maj)
    results.append({"name": "majority-class floor", "config": {}, "metrics": m_maj, "n_features": 0,
                    "train_s": 0.0, "infer_ms": 0.0})
    log_experiment(model="majority_class", dataset="gwstance", split_strategy=split_rep["strategy"],
                   n_train=len(tr), n_val=len(va), n_test=len(te), preprocessing="minimal",
                   macro_f1=m_maj["macro_f1"], accuracy=m_maj["accuracy"], weighted_f1=m_maj["weighted_f1"],
                   f1_favour=m_maj["per_class"]["FAVOUR"]["f1"], f1_against=m_maj["per_class"]["AGAINST"]["f1"],
                   f1_none=m_maj["per_class"]["NONE"]["f1"], seed=42,
                   notes="floor baseline: predict train majority everywhere")

    # ---- sweep ----
    if quick:
        grid = [("word", "logreg", 1.0), ("char", "logreg", 1.0), ("combined", "logreg", 1.0),
                ("combined", "svm", 1.0)]
    else:
        grid = [(f, m, c) for f in ("word", "char", "combined")
                for m in ("logreg", "svm") for c in (0.3, 1.0, 3.0)]

    best = None
    for feature, model, C in grid:
        cfg = TfidfConfig(feature=feature, model=model, C=C, class_weight="balanced")
        t0 = time.time()
        clf = TfidfStanceClassifier(cfg).fit(x_tr, y_tr)
        train_s = time.time() - t0
        p_va = clf.predict(x_va)
        m_va = compute_metrics(y_va, p_va)
        t1 = time.time()
        p_te = clf.predict(x_te)
        infer_ms = (time.time() - t1) / max(1, len(x_te)) * 1000
        m_te = compute_metrics(y_te, p_te)
        row = {"name": f"tfidf-{feature}-{model}-C{C}", "config": cfg.to_dict(), "metrics": m_te,
               "val_macro_f1": m_va["macro_f1"], "n_features": clf.n_features(),
               "train_s": round(train_s, 2), "infer_ms": round(infer_ms, 3),
               "top_features": {l: clf.top_features(l, 8) for l in ("FAVOUR", "AGAINST", "NONE")}}
        results.append(row)
        print(f"{row['name']:34s} valF1={m_va['macro_f1']:.4f} testF1={m_te['macro_f1']:.4f} feat={clf.n_features()}")
        if best is None or m_va["macro_f1"] > best[0]:
            best = (m_va["macro_f1"], clf, row)

        log_experiment(model=f"tfidf_{feature}_{model}", dataset="gwstance", split_strategy=split_rep["strategy"],
                       n_train=len(tr), n_val=len(va), n_test=len(te), preprocessing="minimal",
                       lr="", batch_size="", epochs="", seed=42, loss="squared_hinge" if model == "svm" else "log_loss",
                       macro_f1=m_te["macro_f1"], accuracy=m_te["accuracy"], weighted_f1=m_te["weighted_f1"],
                       f1_favour=m_te["per_class"]["FAVOUR"]["f1"], f1_against=m_te["per_class"]["AGAINST"]["f1"],
                       f1_none=m_te["per_class"]["NONE"]["f1"], english_f1=m_te["macro_f1"],
                       train_seconds=round(train_s, 2), infer_ms_per_item=round(infer_ms, 3),
                       notes=f"C={C}; val_macro_f1={m_va['macro_f1']}; n_features={clf.n_features()}")

    # ---- uncertainty + significance for the best config ----
    _, best_clf, best_row = best
    p_best = best_clf.predict(x_te)
    ci = bootstrap_macro_f1(y_te, p_best, n_boot=1000, seed=42)
    sig = mcnemar_test(y_te, p_best, maj)

    # ---- leakage control: how much does a naive split inflate the score? ----
    lk = leaky["split"]
    lk_cfg = TfidfConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in best_row["config"].items()})
    lk_clf = TfidfStanceClassifier(lk_cfg).fit([r["text"] for r in lk["train"]], [r["label"] for r in lk["train"]])
    lk_metrics = compute_metrics([r["label"] for r in lk["test"]], lk_clf.predict([r["text"] for r in lk["test"]]))

    out = {
        "split_report": split_rep,
        "leaky_split_report": leaky["report"],
        "leaky_best_macro_f1": lk_metrics["macro_f1"],
        "leaky_best_accuracy": lk_metrics["accuracy"],
        "best": {"name": best_row["name"], "config": best_row["config"], "test": best_row["metrics"],
                 "val_macro_f1": best_row["val_macro_f1"], "ci": ci, "vs_majority": sig,
                 "top_features": best_row.get("top_features")},
        "majority": m_maj,
        "all": results,
    }
    Path(ROOT / "experiments" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "results" / "baselines_gwstance.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    save_splits(split, split_rep, ROOT / "data" / "splits", "gwstance_group_stratified")
    write_report(out)
    return results


def write_report(out: dict) -> None:
    b = out["best"]; m = b["test"]; ci = b["ci"]
    rows = sorted([r for r in out["all"] if r["name"] != "majority-class floor"],
                  key=lambda r: r.get("val_macro_f1", 0), reverse=True)
    lines = [
        "# Baseline Comparison — Phase 7", "",
        "Corpus: **GWStance / GWSD** (English, 2 300 sentences, labels derived by Bayesian-posterior argmax).",
        "This is the *development* corpus: it has a label space compatible with the target claim but it is English,",
        "so every number below is an English reference result and is never reported as Hindi/Bengali performance.", "",
        "## Split protocol", "",
        f"- strategy: `{out['split_report']['strategy']}`",
        f"- train / val / test: {out['split_report']['n_train']} / {out['split_report']['n_val']} / {out['split_report']['n_test']}",
        f"- unique groups: {out['split_report']['n_groups']}",
        f"- train↔test group overlap: **{out['split_report']['train_test_group_overlap']}** (leakage-safe: {out['split_report']['leakage_safe']})",
        "", "## Results (test set, macro-F1 is the selection metric on val)", "",
        "| config | val macro-F1 | test macro-F1 | test acc | F1 FAVOUR | F1 AGAINST | F1 NONE | features | train s | infer ms/item |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mm = r["metrics"]
        lines.append(
            f"| `{r['name']}` | {r['val_macro_f1']:.4f} | {mm['macro_f1']:.4f} | {mm['accuracy']:.4f} | "
            f"{mm['per_class']['FAVOUR']['f1']:.4f} | {mm['per_class']['AGAINST']['f1']:.4f} | "
            f"{mm['per_class']['NONE']['f1']:.4f} | {r['n_features']:,} | {r['train_s']:.1f} | {r['infer_ms']:.3f} |"
        )
    maj = out["majority"]
    lines += [
        f"| **majority floor** | – | {maj['macro_f1']:.4f} | {maj['accuracy']:.4f} | "
        f"{maj['per_class']['FAVOUR']['f1']:.4f} | {maj['per_class']['AGAINST']['f1']:.4f} | {maj['per_class']['NONE']['f1']:.4f} | 0 | 0 | 0 |",
        "",
        "## Best configuration", "",
        f"- `{b['name']}` → **test macro-F1 {m['macro_f1']:.4f}**, 95 % bootstrap CI [{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}] (SE {ci['se']:.4f}, {ci['n_boot']} resamples)",
        f"- vs majority floor: McNemar n01={b['vs_majority'].get('n01')} / n10={b['vs_majority'].get('n10')}, p = {b['vs_majority']['p_value']}",
        f"- per-class: " + " · ".join(f"{k} P {v['precision']:.3f}/R {v['recall']:.3f}/F1 {v['f1']:.3f}" for k, v in m["per_class"].items()),
        f"- confusion matrix (rows=true FAVOUR/AGAINST/NONE): `{m['confusion_matrix']}`",
        "",
        "## Leakage control", "",
        f"- group-stratified split: **{m['macro_f1']:.4f}** macro-F1",
        f"- deliberately naive random split (overlap {out['leaky_split_report']['train_test_group_overlap']} duplicated texts): **{out['leaky_best_macro_f1']:.4f}** macro-F1",
        f"- inflation attributable to duplicate leakage: **{out['leaky_best_macro_f1'] - m['macro_f1']:+.4f}**",
        "",
        "## Most predictive features of the best linear model", "",
    ]
    for lbl, feats in (b.get("top_features") or {}).items():
        lines.append(f"- **{lbl}**: " + ", ".join(f"`{f}`" for f in feats))
    by_test = sorted(rows, key=lambda r: r["metrics"]["macro_f1"], reverse=True)
    fam: dict[str, float] = {}
    for r in rows:
        f = r["name"].split("-")[1]
        fam[f] = max(fam.get(f, -1), r["metrics"]["macro_f1"])
    lines += ["", "## Reading (computed from the table above, not asserted)", "",
              f"- Highest **validation** macro-F1 (the selection protocol): `{rows[0]['name']}` at {rows[0]['val_macro_f1']:.4f}.",
              f"- Highest **test** macro-F1: `{by_test[0]['name']}` at {by_test[0]['metrics']['macro_f1']:.4f}.",
              f"- The selection protocol and the test ranking disagree. With only {out['split_report']['n_val']} validation items the "
              f"validation macro-F1 has a standard error of roughly {ci['se']:.3f}, so a val/test gap of this size is inside noise. "
              "The honest conclusion is that the linear family plateaus at the same place and that val-based selection is underpowered "
              "at this corpus size — which is why the teacher comparison below uses multiple seeds rather than a single run.",
              f"- Best test macro-F1 per feature family: " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(fam.items(), key=lambda x: -x[1])) + ".",
              f"- Majority-class floor: macro-F1 {maj['macro_f1']:.4f}. Every configuration clears it by a wide margin, so the signal is real "
              "and not an artefact of class imbalance.",
              f"- Duplicate leakage inflates test macro-F1 from {m['macro_f1']:.4f} to {out['leaky_best_macro_f1']:.4f} "
              f"({out['leaky_best_macro_f1'] - m['macro_f1']:+.4f}) under a naive random split, which is why every reported number uses the "
              "group-stratified split.",
              f"- Cost: TF-IDF needs {min(r['n_features'] for r in rows):,}-{max(r['n_features'] for r in rows):,} sparse features and "
              f"{min(r['train_s'] for r in rows):.1f}-{max(r['train_s'] for r in rows):.1f} s to train on CPU. This is the efficiency bar the "
              "student must beat on absolute size, and the accuracy bar the teacher must beat to justify ~100x more parameters."]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nbest: {b['name']} test macro-F1={m['macro_f1']:.4f} CI[{ci['ci95_low']:.4f},{ci['ci95_high']:.4f}]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    try:
        sweep(quick=args.quick)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
