"""Cross-lingual transfer probe.

CONSTRAINT: this project has no Hindi/Bengali *stance* labels yet, so a true
cross-lingual stance experiment cannot be run. What can be run - and is clearly
labelled as such - is a Hindi->Bengali *sentiment* transfer on IndicSentiment,
which tests the transfer machinery (shared encoder, zero-shot direction) on
genuine Devanagari and Bengali bytes without pretending it measures stance.

Usage: python -m src.evaluation.cross_lingual
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(2)

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.models.teacher import StanceModel, load_tokenizer  # noqa: E402
from src.training.trainer import TrainConfig, train_model  # noqa: E402
from src.utils.registry import log_experiment  # noqa: E402

SENT_LABELS = ["Positive", "Negative"]
SENT2ID = {l: i for i, l in enumerate(SENT_LABELS)}


def load_sentiment(lang: str) -> list[dict]:
    rows = []
    for split in ("validation", "test"):
        f = ROOT / "data" / "raw" / "indic_sentiment" / f"{split}_{lang}.json"
        if not f.exists() or f.stat().st_size < 100:
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            txt = (r.get("INDIC REVIEW") or "").strip()
            lbl = (r.get("LABEL") or "").strip()
            if txt and lbl in SENT2ID:
                rows.append({"text": txt, "label": lbl, "split": split})
    return rows


class SentDataset(Dataset):
    def __init__(self, rows, tok, max_length=64):
        self.rows, self.tok, self.ml = rows, tok, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(str(r["text"]), truncation=True, max_length=self.ml,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(SENT2ID[r["label"]], dtype=torch.long)
        return item


def evaluate(model, rows, tok) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    dl = DataLoader(SentDataset(rows, tok), batch_size=16)
    preds, golds = [], []
    with torch.no_grad():
        for b in dl:
            y = b.pop("labels")
            logits = model(**b)["logits"]
            preds.extend(logits.argmax(-1).tolist())
            golds.extend(y.tolist())
    y_t = [SENT_LABELS[i] for i in golds]
    y_p = [SENT_LABELS[i] for i in preds]
    return {"n": len(y_t), "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
            "macro_f1": round(float(f1_score(y_t, y_p, average="macro", zero_division=0)), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args(argv)

    hi, bn = load_sentiment("hi"), load_sentiment("bn")
    print(f"sentiment corpus: hi={len(hi)} bn={len(bn)} (2 classes; NOT stance)")
    tok = load_tokenizer("google/muril-base-cased")

    # depth-reduced encoder, same recipe as the stance student
    model = StanceModel("google/muril-base-cased", num_labels=2, pooling="mean", num_hidden_layers=4)
    cfg = TrainConfig(lr=3e-5, epochs=2, batch_size=4, grad_accum=2, max_steps=args.max_steps,
                      use_class_weights=True, seed=42, log_every=50)
    hi_train = [r for r in hi if r["split"] == "validation"]
    hi_test = [r for r in hi if r["split"] == "test"]
    model, hist = train_model(model, hi_train, hi_test, tok, cfg, verbose=True)

    res = {
        "protocol": "hi->bn zero-shot sentiment transfer (proxy for transfer machinery only)",
        "note": "IndicSentiment is translated product reviews; this is SENTIMENT, not stance. "
                "It validates the cross-lingual path, not the project's headline claim.",
        "encoder": "google/muril-base-cased", "layers": 4, "pooling": "mean",
        "n_train_hi": len(hi_train), "n_test_hi": len(hi_test), "n_test_bn": len(bn),
        "in_language_hi": evaluate(model, hi_test, tok),
        "zero_shot_bn": evaluate(model, bn, tok),
        "steps": hist.steps,
    }
    res["transfer_gap"] = round(res["in_language_hi"]["macro_f1"] - res["zero_shot_bn"]["macro_f1"], 4)
    p = ROOT / "experiments" / "results" / "cross_lingual_sentiment_probe.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log_experiment(model="muril-4L (sentiment probe)", dataset="IndicSentiment hi->bn",
                   split_strategy="language_held_out", n_train=len(hi_train), n_val=0,
                   n_test=len(bn), preprocessing="minimal", loss="ce_focal=None",
                   lr=3e-5, batch_size=4, epochs=2, seed=42,
                   macro_f1=res["zero_shot_bn"]["macro_f1"], accuracy=res["zero_shot_bn"]["accuracy"],
                   hindi_f1=res["in_language_hi"]["macro_f1"], bengali_f1=res["zero_shot_bn"]["macro_f1"],
                   notes="SENTIMENT transfer probe, not stance; hi->bn zero-shot")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
