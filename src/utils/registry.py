"""Permanent, append-only experiment registry.

Every experiment appends one row; nothing is ever overwritten. The CSV is the
single source of truth for the comparison tables in `reports/`.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments" / "experiment_registry.csv"

FIELDS = [
    "exp_id", "timestamp_utc", "model", "model_params", "model_size_mb",
    "dataset", "dataset_combo", "split_strategy", "n_train", "n_val", "n_test",
    "preprocessing", "augmentation", "loss", "lr", "batch_size", "epochs", "seed",
    "distill_T", "distill_alpha", "distill_beta", "peft",
    "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
    "accuracy", "weighted_f1", "f1_favour", "f1_against", "f1_none",
    "hindi_f1", "bengali_f1", "english_f1",
    "train_seconds", "infer_ms_per_item", "peak_mem_mb", "notes",
]


def _next_id() -> str:
    if not REGISTRY.exists():
        return "EXP-0001"
    with REGISTRY.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return f"EXP-{len(rows) + 1:04d}"


def log_experiment(**kwargs) -> dict:
    """Append one experiment row. Unknown keys are kept in `notes` as JSON."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    row = {f: kwargs.pop(f, "") for f in FIELDS if f != "exp_id"}
    row["exp_id"] = kwargs.pop("exp_id", None) or _next_id()
    row["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if kwargs:
        extra = json.dumps(kwargs, ensure_ascii=False, default=str)
        row["notes"] = f"{row['notes']} | extra={extra}".strip(" |")

    new_file = not REGISTRY.exists()
    with REGISTRY.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return row


def read_registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    with REGISTRY.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def registry_markdown(sort_by: str = "macro_f1") -> str:
    rows = read_registry()
    if not rows:
        return "_registry empty_"
    def key(r):
        try:
            return float(r.get(sort_by) or 0)
        except ValueError:
            return 0.0
    rows = sorted(rows, key=key, reverse=True)
    cols = ["exp_id", "model", "dataset", "split_strategy", "macro_f1", "accuracy", "f1_favour", "f1_against", "f1_none", "notes"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, ""))[:60] for c in cols) + " |")
    return "\n".join(out)
