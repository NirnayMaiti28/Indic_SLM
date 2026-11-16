"""Memory/throughput feasibility probe for candidate encoders on this CPU box.

The brief says do not assume unlimited GPU resources; here there is no GPU at
all and only ~1.9 GB RAM. This probe measures what actually fits before any
training script is written, so that the teacher/student choice is an empirical
result rather than a hope.

Usage: python -m src.utils.probe_memory
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None
    import psutil

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]

CANDIDATES = [
    ("google/muril-base-cased", 236_000_000),
    ("ai4bharat/IndicBERTv2-MLM-only", 278_000_000),
    ("distilbert-base-multilingual-cased", 134_000_000),
    ("xlm-roberta-base", 278_000_000),
]


def peak_mb() -> float:
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def probe(model_name: str, batch: int = 2, seq: int = 64, lora: bool = False, backward: bool = True) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    rec: dict = {"model": model_name, "batch": batch, "seq": seq, "lora": lora, "backward": backward}
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        if lora:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                             target_modules=["query", "value"])
            model = get_peft_model(model, cfg)
            rec["trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        rec["total_params"] = int(sum(p.numel() for p in model.parameters()))
        rec["load_seconds"] = round(time.time() - t0, 1)
        rec["peak_mb_after_load"] = round(peak_mb(), 1)

        model.train()
        enc = tok(["जलवायु परिवर्तन एक गंभीर समस्या है"] * batch, return_tensors="pt",
                  padding=True, truncation=True, max_length=seq)
        out = model(**enc, labels=torch.zeros(batch, dtype=torch.long))
        loss = out.loss
        t1 = time.time()
        if backward:
            loss.backward()
        step_s = time.time() - t1
        rec.update({"step_seconds": round(step_s, 2),
                    "peak_mb_after_step": round(peak_mb(), 1),
                    "status": "ok"})
        del model, out, loss
        import gc

        gc.collect()
    except Exception as exc:
        rec.update({"status": "fail", "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "peak_mb_after_step": round(peak_mb(), 1)})
    return rec


def main() -> int:
    import gc

    import torch

    torch.set_num_threads(2)
    results = []
    print(f"baseline RSS after torch import: {peak_mb():.0f} MB")
    for name, _params in CANDIDATES:
        for lora in (False, True):
            r = probe(name, batch=2, seq=64, lora=lora)
            results.append(r)
            print(json.dumps(r, ensure_ascii=False))
            gc.collect()
    out = ROOT / "experiments" / "results" / "memory_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
