"""Efficiency benchmarking: parameters, storage, latency percentiles, throughput,
peak memory, and INT8 dynamic quantization - all measured, none assumed.

Usage: python -m src.evaluation.efficiency
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None
    import psutil

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(2)

from src.models.teacher import load_stance_model, load_tokenizer  # noqa: E402

SENTENCES = [
    "जलवायु परिवर्तन एक गंभीर समस्या है।",
    "গ্লোবাল ওয়ার্মিং একটি গুরুতর উদ্বেগের বিষয়।",
    "ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है।",
    "জলবায়ু পরিবর্তন কোনো গুরুতর সমস্যা নয়।",
    "climate change koi serious problem nahi hai",
]


def peak_mb() -> float:
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def dir_size_mb(p: Path) -> float:
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 2) if p.exists() else 0.0


def bench(model, tok, n: int = 100, batch: int = 1, max_length: int = 64) -> dict:
    model.eval()
    texts = [SENTENCES[i % len(SENTENCES)] for i in range(n)]
    lat = []
    with torch.no_grad():
        for i in range(n):
            enc = tok(texts[i], truncation=True, max_length=max_length, return_tensors="pt")
            t0 = time.perf_counter()
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    total = time.perf_counter() - t0
    return {
        "n": n, "batch_size": batch,
        "mean_ms": round(statistics.mean(lat), 2),
        "p50_ms": round(lat[len(lat) // 2], 2),
        "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 2),
        "min_ms": round(lat[0], 2), "max_ms": round(lat[-1], 2),
        "throughput_items_per_s": round(n / total, 2),
    }


def measure(tag: str, model_dir: str, pooling: str, layers: int | None, quantize: bool = False) -> dict:
    tok = load_tokenizer("google/muril-base-cased")
    before = peak_mb()
    model = load_stance_model(model_dir, pooling=pooling, num_hidden_layers=layers)
    after = peak_mb()
    n_params = int(sum(p.numel() for p in model.parameters()))

    rec = {
        "tag": tag, "model_dir": model_dir, "layers": layers, "quantized": quantize,
        "params": n_params, "params_millions": round(n_params / 1e6, 1),
        "storage_mb": dir_size_mb(Path(model_dir)),
        "peak_rss_mb_after_load": round(after, 1),
        "rss_delta_mb": round(after - before, 1),
    }
    rec["latency_b1"] = bench(model, tok, n=60, batch=1)
    rec["latency_b8"] = bench(model, tok, n=64, batch=8)
    rec["peak_rss_mb_after_bench"] = round(peak_mb(), 1)

    if quantize:
        try:
            q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            q.eval()
            rec["int8"] = {
                "latency_b1": bench(q, tok, n=60, batch=1),
                "peak_rss_mb": round(peak_mb(), 1),
                "status": "ok",
            }
        except Exception as exc:
            rec["int8"] = {"status": "fail", "error": f"{type(exc).__name__}: {str(exc)[:150]}"}
    print(f"{tag:12s} params={rec['params_millions']}M storage={rec['storage_mb']}MB "
          f"p50={rec['latency_b1']['p50_ms']}ms p95={rec['latency_b1']['p95_ms']}ms "
          f"thr_b8={rec['latency_b8']['throughput_items_per_s']}/s")
    return rec


def main(argv=None) -> int:
    import argparse
    import gc

    ap = argparse.ArgumentParser()
    ap.add_argument("--student-layers", type=int, default=4)
    args = ap.parse_args(argv)

    results = []
    t = measure("teacher", str(ROOT / "models" / "teacher"), "cls", None, quantize=True)
    results.append(t)
    gc.collect()
    s = measure("student", str(ROOT / "models" / "student"), "mean", args.student_layers, quantize=True)
    results.append(s)

    summary = {
        "teacher": {"params_millions": t["params_millions"], "storage_mb": t["storage_mb"],
                    "p50_ms": t["latency_b1"]["p50_ms"], "p95_ms": t["latency_b1"]["p95_ms"],
                    "throughput_b8": t["latency_b8"]["throughput_items_per_s"],
                    "peak_rss_mb": t["peak_rss_mb_after_bench"]},
        "student": {"params_millions": s["params_millions"], "storage_mb": s["storage_mb"],
                    "p50_ms": s["latency_b1"]["p50_ms"], "p95_ms": s["latency_b1"]["p95_ms"],
                    "throughput_b8": s["latency_b8"]["throughput_items_per_s"],
                    "peak_rss_mb": s["peak_rss_mb_after_bench"]},
    }
    summary["compression"] = {
        "param_ratio": round(t["params"] / s["params"], 2),
        "storage_ratio": round(t["storage_mb"] / s["storage_mb"], 2) if s["storage_mb"] else None,
        "p50_speedup": round(t["latency_b1"]["p50_ms"] / s["latency_b1"]["p50_ms"], 2) if s["latency_b1"]["p50_ms"] else None,
        "throughput_gain": round(s["latency_b8"]["throughput_items_per_s"] / t["latency_b8"]["throughput_items_per_s"], 2),
    }
    out = {"runs": results, "summary": summary}
    p = ROOT / "experiments" / "results" / "efficiency.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
