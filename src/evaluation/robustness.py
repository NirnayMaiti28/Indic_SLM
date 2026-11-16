"""Robustness evaluation: how much macro-F1 survives realistic social-media noise.

Every perturbation is applied to the *test split only* and is label-preserving by
construction. Because none of these operations has a stance value, the label
stays valid; that assumption is stated explicitly and is the reason no
semantic rewrites (paraphrase/back-translation) are used here - those can flip
stance and would require re-annotation.

Usage: python -m src.evaluation.robustness
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(2)

from torch.utils.data import DataLoader  # noqa: E402

from src.data.augmentation import (  # noqa: E402
    char_repeat, emoji_append, english_codemix, hashtag_append, partial_romanize, spacing_noise,
)
from src.data.dataset_loader import load_gwstance  # noqa: E402
from src.data.splitting import group_stratified_split  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.teacher import load_stance_model, load_tokenizer  # noqa: E402
from src.training.trainer import StanceDataset, evaluate_model  # noqa: E402

PERTURBATIONS = {
    "clean": lambda t, r: t,
    "char_elongation": lambda t, r: char_repeat(t, 0.35, r),
    "spacing_noise": lambda t, r: spacing_noise(t, 0.30, r),
    "emoji_suffix": lambda t, r: emoji_append(t, rng=r),
    "hashtag_suffix": lambda t, r: hashtag_append(t, rng=r),
    "partial_romanization": lambda t, r: partial_romanize(t, 0.6, r),
    "english_codemix": lambda t, r: english_codemix(t, rng=r),
    "all_noise": lambda t, r: partial_romanize(
        emoji_append(hashtag_append(spacing_noise(char_repeat(t, 0.3, r), 0.25, r), rng=r), rng=r), 0.5, r),
}


def run(model_dir: str, pooling: str, layers: int | None, tag: str) -> dict:
    split, _ = group_stratified_split(load_gwstance(drop_screening=True), seed=42)
    te = split["test"]
    tok = load_tokenizer("google/muril-base-cased")
    model = load_stance_model(model_dir, pooling=pooling, num_hidden_layers=layers)
    model.eval()

    results = {}
    for name, fn in PERTURBATIONS.items():
        rng = random.Random(7)
        rows = [{"text": fn(str(r["text"]), rng), "label": r["label"]} for r in te]
        dl = DataLoader(StanceDataset(rows, tok, 64), batch_size=16, shuffle=False)
        ev = evaluate_model(model, dl)
        m = compute_metrics(ev["y_true"], ev["y_pred"])
        results[name] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                         "per_class": m["per_class"], "n": m["n"]}
        print(f"{tag:10s} {name:22s} macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")

    clean = results["clean"]["macro_f1"]
    for k, v in results.items():
        v["delta_vs_clean"] = round(v["macro_f1"] - clean, 4)
        v["retention"] = round(v["macro_f1"] / clean, 4) if clean else 0.0

    out = {"tag": tag, "model_dir": model_dir, "n_test": len(te), "results": results,
           "mean_retention": round(sum(v["retention"] for v in results.values()) / len(results), 4)}
    p = ROOT / "experiments" / "results" / f"robustness_{tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{tag}: mean retention {out['mean_retention']:.4f} -> {p}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "student"))
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tag", default="student")
    args = ap.parse_args(argv)
    run(args.model_dir, args.pooling, args.layers, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
