"""
End-to-End Multilingual Verification Suite for Indic Climate Stance SLM Platform.
Simulates all frontend UI workflows against the live production server (port 8000):
- Hindi, Bengali, Hinglish, Banglish, Neutral stance predictions
- Negation detection & code-mixing linguistic profiling
- Integrated Gradients token attribution extraction
- Batch analysis & CSV upload/export pipeline
- Benchmark evaluation matrix verification (Top model 0.9916 vs Deployed SLM 0.9574)
- Frontend static asset bundle delivery
"""

import json
import urllib.request
import urllib.parse
import io
import time

BASE_URL = "http://127.0.0.1:8000"

def post_json(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(endpoint: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_raw(path: str):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.headers, resp.read()

def post_multipart_csv(endpoint: str, csv_content: str, filename="dataset.csv") -> dict:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.write(b"Content-Type: text/csv\r\n\r\n")
    body.write(csv_content.encode("utf-8"))
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    data = body.getvalue()
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("=" * 80)
    print("🧪 RUNNING END-TO-END APPLICATION & API WORKFLOW TEST SUITE")
    print("=" * 80)

    # 1. Server Health Check
    print("\n[TEST 1] Verifying System Health & Model Status (/api/v1/health)...")
    health = get_json("/api/v1/health")
    print(f"  • Status:     {health['status']}")
    print(f"  • Model:      {health['model_name']}")
    print(f"  • Device:     {health['device']}")
    print(f"  • Checkpoint: {health['checkpoint']}")
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True

    # 2. Multilingual Stance Analyzer Tests (Hindi, Bengali, Hinglish, Banglish)
    print("\n[TEST 2] Testing Multilingual Stance Analyzer (/api/v1/predict)...")
    test_cases = [
        {
            "category": "Hindi Concern (FAVOUR)",
            "text": "जलवायु परिवर्तन हमारे अस्तित्व के लिए सबसे गंभीर खतरा है।",
            "expected_stance": "FAVOUR",
            "expected_lang": "Hindi",
            "expected_script": "Devanagari"
        },
        {
            "category": "Bengali Concern (FAVOUR)",
            "text": "জলবায়ু পরিবর্তন একটি বাস্তব ও ভয়াবহ সংকট, এখনই কার্বন নির্গমন কমাতে হবে। 🌍",
            "expected_stance": "FAVOUR",
            "expected_lang": "Bengali",
            "expected_script": "Bengali"
        },
        {
            "category": "Bengali Negation Denial (AGAINST)",
            "text": "গ্লোবাল ওয়ার্মিং কোনো গুরুতর সমস্যা নয়, এটি একটি সাজানো নাটক।",
            "expected_stance": "AGAINST",
            "expected_lang": "Bengali",
            "expected_script": "Bengali",
            "has_negation": True
        },
        {
            "category": "Hinglish Code-Mixed (FAVOUR)",
            "text": "Climate change sach me ek bahut serious concern hai bhai, glaciers melt ho rahe hain.",
            "expected_stance": "FAVOUR",
            "expected_lang": "Hindi",
            "expected_script": "Latin"
        },
        {
            "category": "Banglish Denial (AGAINST)",
            "text": "Global warming kono serious issue na, shob funding pabar natok.",
            "expected_stance": "AGAINST",
            "expected_lang": "Bengali",
            "expected_script": "Latin",
            "has_negation": True
        },
        {
            "category": "Neutral Inquiry (NONE)",
            "text": "सौर ऊर्जा पैनल लगाने का सरकारी खर्च कितना आता है, कोई जानकारी दे सकता है?",
            "expected_stance": "NONE",
            "expected_lang": "Hindi",
            "expected_script": "Devanagari"
        }
    ]

    for tc in test_cases:
        res = post_json("/api/v1/predict", {"text": tc["text"], "compute_attributions": False})
        print(f"\n  • [{tc['category']}]")
        print(f"    Input:       \"{tc['text'][:65]}...\"")
        print(f"    Stance:      {res['predicted_label']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"    Probs:       FAVOUR: {res['probabilities']['FAVOUR']:.3f} | AGAINST: {res['probabilities']['AGAINST']:.3f} | NONE: {res['probabilities']['NONE']:.3f}")
        print(f"    Profile:     Lang={res['language']}, Script={res['script']}, Negation={res['has_negation']}, Latency={res['latency_ms']:.2f}ms")
        
        assert res["predicted_label"] == tc["expected_stance"], f"Expected {tc['expected_stance']}, got {res['predicted_label']}"
        assert res["script"] == tc["expected_script"]
        if "has_negation" in tc:
            assert res["has_negation"] == tc["has_negation"]

    # 3. Explainability: Integrated Gradients Token Attributions
    print("\n[TEST 3] Testing Model Explainability (/api/v1/predict with explain=True)...")
    explain_res = post_json("/api/v1/predict", {
        "text": "गंभीर जलवायु खतरा",
        "explain": True
    })
    print(f"  • Predicted stance: {explain_res['predicted_label']}")
    explanations = explain_res.get("explanations") or []
    print(f"  • Tokens analyzed:  {len(explanations)} tokens")
    assert len(explanations) > 0
    top_tokens = [(e["token"], round(e["attribution"], 4)) for e in explanations]
    print(f"  • Token Attribution Scores: {top_tokens}")

    # 4. Batch Analysis & CSV Pipeline
    print("\n[TEST 4] Testing Batch Analysis & CSV Upload/Download Pipeline (/api/v1/predict/file)...")
    csv_sample = '''id,text
1,"जलवायु परिवर्तन हमारे अस्तित्व के लिए सबसे गंभीर खतरा है।"
2,"গ্লোবাল ওয়ার্মিং কোনো গুরুতর সমস্যা নয়, এটি একটি সাজানো নাটক।"
3,"Climate change sach me ek bahut serious concern hai bhai."
4,"Global warming kono serious issue na, shob funding pabar natok."
5,"सौर ऊर्जा पैनल लगाने का सरकारी खर्च कितना आता है?"
'''
    file_res = post_multipart_csv("/api/v1/predict/file?text_column=text", csv_sample)
    print(f"  • Filename:         {file_res['filename']}")
    print(f"  • Processed rows:   {file_res['total_records']}")
    print(f"  • Class breakdown:  {file_res['class_distribution']}")
    print(f"  • Batch throughput: {file_res['throughput_items_per_sec']:.2f} records/sec")
    for r in file_res["records"]:
        print(f"    - ID {r['id']}: stance={r['predicted_stance']}, conf={r['confidence']:.3f}, text={r['text'][:40]}")
    assert file_res["total_records"] == 5
    assert file_res["class_distribution"]["FAVOUR"] >= 1
    assert file_res["class_distribution"]["AGAINST"] >= 1
    assert file_res["class_distribution"]["NONE"] >= 1

    # Verify CSV export reconstruction
    records = file_res["records"]
    assert len(records) == 5
    print(f"  • Sample exported record: id={records[0]['id']}, stance={records[0]['predicted_stance']}, conf={records[0]['confidence']:.3f}")

    # 5. Research Benchmarks & Model Ranking
    print("\n[TEST 5] Verifying Research Benchmarks & Highest Scoring Model (/api/v1/benchmarks)...")
    bench = get_json("/api/v1/benchmarks")
    models = bench["models"]
    print(f"  • Total models evaluated: {len(models)}")
    
    # Calculate top model
    highest_overall = max(models, key=lambda m: m["macro_f1"])
    deployed_slm = next(m for m in models if "Distilled" in m["name"])
    
    print(f"  • Highest overall benchmark:  {highest_overall['name']} (Macro-F1: {highest_overall['macro_f1']})")
    print(f"  • Deployed Neural SLM:         {deployed_slm['name']} (Macro-F1: {deployed_slm['macro_f1']})")
    
    assert "TF-IDF Char" in highest_overall["name"]
    assert highest_overall["macro_f1"] == 0.9916
    assert deployed_slm["macro_f1"] == 0.9574
    print("  ✅ Dashboard correctly identifies TF-IDF Char + Linear SVM (0.9916) as the highest-scoring benchmark!")

    # 6. Model Information Endpoint
    print("\n[TEST 6] Verifying Technical Model Info (/api/v1/model/info)...")
    info = get_json("/api/v1/model/info")
    print(f"  • Base model:       {info['base_encoder']}")
    print(f"  • Encoder layers:   {info['layers']} (pruned from 12)")
    print(f"  • LoRA rank:        {info['lora_rank']} (alpha={info['lora_alpha']})")
    print(f"  • Total parameters: {info['total_parameters']:,}")
    print(f"  • Trainable params: {info['trainable_parameters']:,}")
    assert info["layers"] == 4
    assert info["lora_rank"] == 8
    assert info["trainable_parameters"] == 100611

    # 7. Frontend Static Web Bundle Delivery
    print("\n[TEST 7] Verifying Web SPA Production Bundle Delivery...")
    status, headers, html_bytes = get_raw("/")
    print(f"  • GET /: HTTP {status}, Content-Type={headers.get('content-type')}, Size={len(html_bytes)} bytes")
    assert status == 200
    assert b"Indic Climate Stance" in html_bytes or b"<div id=\"root\">" in html_bytes

    print("\n" + "=" * 80)
    print("🎉 ALL END-TO-END APPLICATION AND API WORKFLOW TESTS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
