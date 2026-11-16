"""FastAPI Application for Indic Climate Stance Detection.

Provides production-ready REST API endpoints for stance classification, explainability, batch file processing, and research benchmarks.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Set up paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    TokenAttribution
)
from backend.benchmarks_data import (
    BENCHMARK_MODELS,
    DATASET_METRICS,
    CONFUSION_MATRIX,
    LATENCY_BENCHMARKS
)
from backend.model_loader import get_inference_engine, StanceInferenceEngine

# Global model engine reference
engine: Optional[StanceInferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to load and cache the model once during backend startup."""
    global engine
    print("[*] Starting Indic Climate Stance Detection API...")
    engine = get_inference_engine()
    print("[+] Model loaded and cached. Server ready to accept requests.")
    yield
    print("[*] Shutting down Indic Climate Stance Detection API...")


app = FastAPI(
    title="Indic Climate Stance Detection API",
    description="Multilingual AI Stance Classification Platform for Hindi and Bengali Social Media Text (Compact Student SLM distilled from MuRIL).",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> StanceInferenceEngine:
    global engine
    if engine is None:
        engine = get_inference_engine()
    return engine


@app.get("/", tags=["General"])
async def root():
    dist_index = ROOT / "frontend" / "dist" / "index.html"
    if dist_index.is_file():
        return FileResponse(dist_index)
    return {
        "message": "Indic Climate Stance Detection API is active.",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check(eng: StanceInferenceEngine = Depends(get_engine)):
    return HealthResponse(
        status="healthy",
        model_loaded=eng._initialized,
        model_name="Compact Student SLM (4-Layer MuRIL + LoRA + KD)",
        checkpoint=str(eng.checkpoint_path.name),
        device=eng.device,
        version="0.1.0"
    )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
async def get_model_info(eng: StanceInferenceEngine = Depends(get_engine)):
    return ModelInfoResponse(
        name="Indic Climate Stance Compact Student SLM",
        architecture="4-Layer Depth-Pruned Transformer + PEFT LoRA (r=8, alpha=16) + Mean-Pooling StanceHead",
        base_encoder=eng.encoder_name,
        layers=4,
        pooling="mean",
        max_sequence_length=64,
        lora_rank=8,
        lora_alpha=16,
        total_parameters=eng.total_params,
        trainable_parameters=eng.trainable_params,
        classes=["FAVOUR", "AGAINST", "NONE"],
        target_claim="Climate change and global warming is a serious concern.",
        supported_languages=["Hindi", "Bengali", "English", "Hinglish (Romanized Hindi)", "Banglish (Romanized Bengali)"],
        supported_scripts=["Devanagari", "Bengali", "Latin", "Mixed"],
        distillation_teacher="google/muril-base-cased (12 layers, 237.8M params)",
        distillation_hyperparameters={
            "temperature": 2.0,
            "alpha_ce": 0.5,
            "beta_kl": 0.5,
            "t_squared_scaling": True
        }
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_single(request: PredictRequest, eng: StanceInferenceEngine = Depends(get_engine)):
    """Classify the climate stance of a single social media comment."""
    res = eng.predict(request.text, explain=request.explain)
    if res.get("error") and res["predicted_label"] is None:
        return PredictResponse(
            text=request.text,
            error=res["error"],
            latency_ms=res.get("latency_ms", 0.0),
            device=eng.device
        )
    return PredictResponse(**res)


@app.post("/api/v1/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest, eng: StanceInferenceEngine = Depends(get_engine)):
    """Batch inference for a list of input texts."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")
    
    batch_out = eng.predict_batch(request.texts, explain=request.explain)
    response_items = [PredictResponse(**r) for r in batch_out["results"]]
    return BatchPredictResponse(
        results=response_items,
        total=batch_out["total"],
        latency_ms=batch_out["latency_ms"],
        throughput_items_per_sec=batch_out["throughput_items_per_sec"]
    )


@app.post("/api/v1/predict/file", tags=["Inference"])
async def predict_file(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    eng: StanceInferenceEngine = Depends(get_engine)
):
    """Upload CSV or JSONL file, classify stances, and return structured predictions with summary."""
    filename = file.filename or ""
    content = await file.read()
    texts: List[str] = []
    original_rows: List[Dict[str, Any]] = []

    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        decoded = content.decode("latin-1")

    if filename.endswith(".csv") or filename.endswith(".tsv"):
        delimiter = "\t" if filename.endswith(".tsv") else ","
        reader = csv.DictReader(io.StringIO(decoded), delimiter=delimiter)
        for r in reader:
            original_rows.append(r)
            texts.append(r.get(text_column, ""))
    elif filename.endswith(".jsonl") or filename.endswith(".ndjson"):
        for line in decoded.splitlines():
            if line.strip():
                item = json.loads(line)
                original_rows.append(item)
                texts.append(item.get(text_column, ""))
    elif filename.endswith(".json"):
        data = json.loads(decoded)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    original_rows.append({"text": item})
                    texts.append(item)
                elif isinstance(item, dict):
                    original_rows.append(item)
                    texts.append(item.get(text_column, ""))
        else:
            raise HTTPException(status_code=400, detail="JSON must be an array of records or strings.")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .csv, .tsv, .json, or .jsonl")

    if not texts:
        raise HTTPException(status_code=400, detail=f"No texts found under column '{text_column}' or file is empty.")

    # Limit maximum batch upload for web API protection
    if len(texts) > 500:
        texts = texts[:500]
        original_rows = original_rows[:500]

    batch_out = eng.predict_batch(texts, explain=False)
    results = batch_out["results"]

    # Build annotated records
    annotated = []
    class_counts = {"FAVOUR": 0, "AGAINST": 0, "NONE": 0}
    lang_counts = {}

    for orig, res in zip(original_rows, results):
        lbl = res.get("predicted_label") or "UNKNOWN"
        lng = res.get("language") or "Unknown"
        if lbl in class_counts:
            class_counts[lbl] += 1
        lang_counts[lng] = lang_counts.get(lng, 0) + 1

        record = {
            **orig,
            "predicted_stance": lbl,
            "confidence": res.get("confidence"),
            "prob_favour": res.get("probabilities", {}).get("FAVOUR") if res.get("probabilities") else 0.0,
            "prob_against": res.get("probabilities", {}).get("AGAINST") if res.get("probabilities") else 0.0,
            "prob_none": res.get("probabilities", {}).get("NONE") if res.get("probabilities") else 0.0,
            "detected_language": lng,
            "detected_script": res.get("script"),
            "code_mixed": res.get("is_code_mixed", False),
            "negation_detected": res.get("has_negation", False)
        }
        annotated.append(record)

    return {
        "filename": filename,
        "total_records": len(annotated),
        "latency_ms": batch_out["latency_ms"],
        "throughput_items_per_sec": batch_out["throughput_items_per_sec"],
        "class_distribution": class_counts,
        "language_distribution": lang_counts,
        "records": annotated
    }


@app.get("/api/v1/benchmarks", tags=["Research Benchmarks"])
async def get_benchmarks():
    """Retrieve full research benchmark results, latency measurements, and confusion matrices."""
    return {
        "models": BENCHMARK_MODELS,
        "datasets": DATASET_METRICS,
        "confusion_matrix": CONFUSION_MATRIX,
        "latency_benchmarks": LATENCY_BENCHMARKS
    }


# Serve built React frontend in production if dist exists
dist_dir = ROOT / "frontend" / "dist"
if dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")

    @app.get("/{full_path:path}", tags=["Frontend SPA"])
    async def serve_spa(full_path: str):
        if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            raise HTTPException(status_code=404, detail="Not Found")
        target_file = dist_dir / full_path
        if target_file.is_file():
            return FileResponse(target_file)
        return FileResponse(dist_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
