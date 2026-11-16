"""Batch Prediction CLI and API for Indic Climate Stance Detection.

Supports:
- CSV input/output
- JSON / JSONL input/output
- Direct text lists
- Streaming batch inference

Usage:
  python -m src.inference.batch_predict --input data/samples.csv --output results.csv
  python -m src.inference.batch_predict --text "जलवायु परिवर्तन गंभीर है" "সব নাটক"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predict import StancePredictor


def predict_texts(texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
    predictor = StancePredictor()
    return predictor.predict_batch(texts, batch_size=batch_size)


def process_csv(in_path: str | Path, out_path: str | Path, text_col: str = "text", batch_size: int = 32):
    in_p, out_p = Path(in_path), Path(out_path)
    rows = []
    with in_p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    texts = [r.get(text_col, "") for r in rows]
    results = predict_texts(texts, batch_size=batch_size)

    for r, res in zip(rows, results):
        r["predicted_label"] = res["predicted_label"]
        r["confidence"] = res["confidence"]
        r["prob_favour"] = res["probabilities"].get("FAVOUR", 0.0)
        r["prob_against"] = res["probabilities"].get("AGAINST", 0.0)
        r["prob_none"] = res["probabilities"].get("NONE", 0.0)
        r["detected_language"] = res.get("language", "Unknown")
        r["detected_script"] = res.get("script", "Unknown")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["predicted_label", "confidence"]
    with out_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Processed {len(rows)} rows from {in_p} -> {out_p}")


def process_jsonl(in_path: str | Path, out_path: str | Path, text_col: str = "text", batch_size: int = 32):
    in_p, out_p = Path(in_path), Path(out_path)
    records = [json.loads(line) for line in in_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [r.get(text_col, "") for r in records]
    results = predict_texts(texts, batch_size=batch_size)

    for r, res in zip(records, results):
        r["prediction"] = res

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[+] Processed {len(records)} records from {in_p} -> {out_p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch inference for Indic Climate Stance")
    ap.add_argument("--input", "-i", type=str, help="Input CSV or JSONL file")
    ap.add_argument("--output", "-o", type=str, help="Output file path")
    ap.add_argument("--text-col", default="text", help="Name of the text column")
    ap.add_argument("--text", nargs="+", help="One or more text strings directly via CLI")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference")
    args = ap.parse_args(argv)

    if args.text:
        results = predict_texts(args.text, batch_size=args.batch_size)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    if not args.input or not args.output:
        ap.print_help()
        return 1

    in_ext = Path(args.input).suffix.lower()
    if in_ext in (".csv", ".tsv"):
        process_csv(args.input, args.output, text_col=args.text_col, batch_size=args.batch_size)
    elif in_ext in (".jsonl", ".json"):
        process_jsonl(args.input, args.output, text_col=args.text_col, batch_size=args.batch_size)
    else:
        print(f"[-] Unsupported file extension: {in_ext}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
