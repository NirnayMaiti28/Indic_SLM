"""Inference API for the deployed student model.

    from src.inference.predict import StancePredictor
    p = StancePredictor()                      # loads once
    p.predict_stance("जलवायु परिवर्तन गंभीर है")
    p.predict_batch([...]); p.predict_csv("in.csv", "out.csv"); p.predict_json(path)

Handles empty/None/over-long/whitespace-only input without raising.
"""
from __future__ import annotations

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

from src.data.preprocessing import detect_script, is_romanized, preprocess  # noqa: E402

LABELS = ["FAVOUR", "AGAINST", "NONE"]


class StancePredictor:
    _instance = None

    def __new__(cls, *a, **kw):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_dir: str | None = None, pooling: str = "mean", layers: int = 4):
        if getattr(self, "_ready", False):
            return
        from src.models.teacher import load_stance_model, load_tokenizer

        self.model_dir = model_dir or str(ROOT / "models" / "student")
        self.tok = load_tokenizer("google/muril-base-cased")
        self.model = load_stance_model(self.model_dir, pooling=pooling, num_hidden_layers=layers)
        self.model.eval()
        self.pooling, self.layers = pooling, layers
        self._ready = True

    def _language(self, text: str) -> tuple[str, str]:
        script = detect_script(text)
        if script == "Devanagari":
            return "Hindi", script
        if script == "Bengali":
            return "Bengali", script
        if script == "Latin":
            return ("Hindi" if is_romanized(text) else "Unknown"), "Latin"
        return "Unknown", script

    @torch.no_grad()
    def predict_stance(self, text: str, explain: bool = False) -> dict:
        if text is None or not str(text).strip():
            return {"text": text, "predicted_label": None, "confidence": None,
                    "probabilities": None, "language": None, "script": None,
                    "error": "empty_input"}
        raw = str(text)
        clean = preprocess(raw, variant="minimal").text
        if not clean.strip():
            return {"text": raw, "predicted_label": None, "confidence": None, "probabilities": None,
                    "language": None, "script": None, "error": "empty_after_preprocessing"}

        enc = self.tok(clean, truncation=True, max_length=64, return_tensors="pt")
        out_model = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out_model["logits"] if isinstance(out_model, dict) else out_model
        probs = torch.softmax(logits, -1)[0]
        idx = int(probs.argmax())
        lang, script = self._language(raw)
        out = {
            "text": raw,
            "predicted_label": LABELS[idx],
            "confidence": round(float(probs[idx]), 4),
            "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(3)},
            "language": lang,
            "script": script,
            "truncated": int(enc["attention_ids"].sum()) > 64 if "attention_ids" in enc else int(enc["attention_mask"].sum()) >= 64,
        }
        if explain:
            from src.evaluation.explainability import integrated_gradients

            ig = integrated_gradients(self.model, self.tok, clean, idx)
            out["explanations"] = [{"token": t, "attribution": a}
                                   for t, a in zip(ig["tokens"][:8], ig["attributions"][:8])]
        return out

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        return [self.predict_stance(t) for t in texts]

    def predict_csv(self, in_path: str, out_path: str, text_col: str = "text") -> dict:
        import csv

        with open(in_path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        preds = [self.predict_stance(r.get(text_col, "")) for r in rows]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["text", "predicted_label", "confidence",
                                               "p_FAVOUR", "p_AGAINST", "p_NONE", "language", "script"])
            w.writeheader()
            for r, p in zip(rows, preds):
                probs = p["probabilities"] or {}
                w.writerow({"text": r.get(text_col, ""), "predicted_label": p["predicted_label"],
                            "confidence": p["confidence"],
                            "p_FAVOUR": probs.get("FAVOUR"), "p_AGAINST": probs.get("AGAINST"),
                            "p_NONE": probs.get("NONE"), "language": p["language"], "script": p["script"]})
        return {"n": len(preds), "out_path": out_path, "errors": sum(1 for p in preds if p.get("error"))}

    def predict_json(self, in_path: str, out_path: str | None = None) -> dict | list:
        data = json.loads(Path(in_path).read_text(encoding="utf-8"))
        texts = data if isinstance(data, list) else [data.get("text", "")]
        preds = self.predict_batch([t if isinstance(t, str) else t.get("text", "") for t in texts])
        if out_path:
            Path(out_path).write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
        return preds


def predict_stance(text: str) -> dict:
    return StancePredictor().predict_stance(text)


if __name__ == "__main__":
    p = StancePredictor()
    for t in ["जलवायु परिवर्तन एक गंभीर समस्या है।",
              "ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है।",
              "জলবায়ু পরিবর্তন একটি গুরুতর উদ্বেগের বিষয়।",
              "climate change koi serious problem nahi hai",
              "", None]:
        print(json.dumps(p.predict_stance(t), ensure_ascii=False))
