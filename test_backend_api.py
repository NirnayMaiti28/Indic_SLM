"""Comprehensive integration test suite for the Indic Climate Stance FastAPI backend."""
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starlette.testclient import TestClient
from backend.main import app

def run_tests():
    print("============================================================")
    print("🚀 RUNNING END-TO-END BACKEND API INTEGRATION TESTS")
    print("============================================================")

    with TestClient(app) as client:
        # 1. Health Endpoint
        print("\n[1] Testing GET /api/v1/health...")
        res = client.get("/api/v1/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        data = res.json()
        print(f"    Status: {data['status']}, Device: {data['device']}, Checkpoint: {data['checkpoint']}")
        assert data["model_loaded"] is True

        # 2. Model Info Endpoint
        print("\n[2] Testing GET /api/v1/model/info...")
        res = client.get("/api/v1/model/info")
        assert res.status_code == 200
        info = res.json()
        print(f"    Architecture: {info['architecture']}")
        print(f"    Params: Total={info['total_parameters']:,}, Trainable={info['trainable_parameters']:,}")
        assert info["layers"] == 4
        assert "FAVOUR" in info["classes"]

        # 3. Single Prediction (Hindi with Integrated Gradients)
        print("\n[3] Testing POST /api/v1/predict (Hindi + Explainability)...")
        req_body = {
            "text": "जलवायु परिवर्तन हमारे अस्तित्व के लिए सबसे गंभीर खतरा है।",
            "explain": True
        }
        res = client.post("/api/v1/predict", json=req_body)
        assert res.status_code == 200
        pred = res.json()
        print(f"    Input:       {pred['text']}")
        print(f"    Predicted:   {pred['predicted_label']} (Confidence: {pred['confidence']*100:.1f}%)")
        print(f"    Probs:       {pred['probabilities']}")
        print(f"    Language:    {pred['language']} | Script: {pred['script']}")
        print(f"    Latency:     {pred['latency_ms']} ms | Device: {pred['device']}")
        print(f"    Top IG tokens: {[t['token'] + '(' + str(t['attribution']) + ')' for t in pred['explanations'][:3]]}")
        assert pred["predicted_label"] == "FAVOUR"
        assert len(pred["explanations"]) > 0

        # 4. Single Prediction (Bengali Negation Denial)
        print("\n[4] Testing POST /api/v1/predict (Bengali Negation Denial)...")
        req_body = {
            "text": "গ্লোবাল ওয়ার্মিং কোনো গুরুতর সমস্যা নয়, এটি একটি সাজানো নাটক।",
            "explain": False
        }
        res = client.post("/api/v1/predict", json=req_body)
        assert res.status_code == 200
        pred = res.json()
        print(f"    Input:       {pred['text']}")
        print(f"    Predicted:   {pred['predicted_label']} (Confidence: {pred['confidence']*100:.1f}%)")
        print(f"    Probs:       {pred['probabilities']}")
        print(f"    Negation:    {pred['has_negation']}")
        assert pred["predicted_label"] == "AGAINST"
        assert pred["has_negation"] is True

        # 5. Batch Prediction
        print("\n[5] Testing POST /api/v1/predict/batch...")
        batch_req = {
            "texts": [
                "जलবায়ু পরিবর্তন একটি বাস্তব সংকট।",
                "Climate change sach me ek bahut serious concern hai bhai.",
                "सौर ऊर्जा लगाने में कितना खर्च आता है?"
            ],
            "explain": False
        }
        res = client.post("/api/v1/predict/batch", json=batch_req)
        assert res.status_code == 200
        b_res = res.json()
        print(f"    Total items: {b_res['total']}")
        print(f"    Latency:     {b_res['latency_ms']} ms ({b_res['throughput_items_per_sec']} items/sec)")
        for i, item in enumerate(b_res["results"]):
            print(f"    [{i+1}] {item['text'][:40]}... -> {item['predicted_label']} ({item['confidence']*100:.1f}%)")
        assert len(b_res["results"]) == 3

        # 6. File Upload Processing
        print("\n[6] Testing POST /api/v1/predict/file (CSV upload)...")
        csv_content = (
            "id,text,source\n"
            "1,ग्लोबल वार्मिंग से ग्लेशियर पिघल रहे हैं,twitter\n"
            "2,यह सब पश्चिमी देशों की साजिश है कोई ग्लोबल वार्मिंग नहीं है,facebook\n"
            "3,weather forecast for today,news\n"
        )
        files = {
            "file": ("test_comments.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
        }
        data = {"text_column": "text"}
        res = client.post("/api/v1/predict/file", files=files, data=data)
        assert res.status_code == 200
        f_res = res.json()
        print(f"    Processed rows: {f_res['total_records']}")
        print(f"    Class distribution: {f_res['class_distribution']}")
        print(f"    Sample row with stance: {f_res['records'][0]['text']} -> {f_res['records'][0]['predicted_stance']}")
        assert f_res["total_records"] == 3

        # 7. Benchmarks Endpoint
        print("\n[7] Testing GET /api/v1/benchmarks...")
        res = client.get("/api/v1/benchmarks")
        assert res.status_code == 200
        bench = res.json()
        print(f"    Models evaluated: {len(bench['models'])}")
        top_model = max(bench["models"], key=lambda m: m["macro_f1"])
        deployed_model = next(m for m in bench["models"] if "Distilled" in m["name"])
        print(f"    Highest-scoring model overall: {top_model['name']} -> {top_model['macro_f1']}")
        print(f"    Deployed neural SLM:           {deployed_model['name']} -> {deployed_model['macro_f1']}")
        assert top_model["macro_f1"] == 0.9916
        assert "TF-IDF Char" in top_model["name"]
        assert deployed_model["macro_f1"] == 0.9574
        assert len(bench["models"]) >= 5

        # 8. Frontend SPA Root Route
        print("\n[8] Testing GET / (Frontend SPA bundle serving)...")
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")
        print("    Frontend HTML bundle served successfully!")

    print("\n============================================================")
    print("✅ ALL 8 END-TO-END BACKEND API INTEGRATION TESTS PASSED!")
    print("============================================================")

if __name__ == "__main__":
    run_tests()

