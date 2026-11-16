import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.language_detection import detect_language, detect_script, analyze_language
from src.data.preprocessing import preprocess


def test_script_detection():
    assert detect_script("जलवायु परिवर्तन") == "Devanagari"
    assert detect_script("জলবায়ু পরিবর্তন") == "Bengali"
    assert detect_script("climate change is real") == "Latin"


def test_language_detection():
    assert detect_language("यह बहुत गंभीर समस्या है") == "Hindi"
    assert detect_language("এটা খুব গুরুতর সমস্যা") == "Bengali"
    assert detect_language("Global warming is real") == "English"
    assert detect_language("bhai warming real hai yaar") == "Hindi"
    assert detect_language("dada climate change asholei serious") == "Bengali"


def test_preprocessing_edge_cases():
    p1 = preprocess("")
    assert p1.text == ""
    assert not p1.flags["has_negation"]

    p2 = preprocess("    ")
    assert p2.text == ""

    p3 = preprocess("ग्लोबल वार्मिंग कोई समस्या नहीं है")
    assert p3.flags["has_negation"]
    assert "नहीं" in p3.text


def test_profile_output_structure():
    prof = analyze_language("ग्लोबल वार्मिंग 🌍🔥")
    d = prof.to_dict()
    assert d["script"] == "Devanagari"
    assert d["language"] == "Hindi"
    assert "script_counts" in d


def test_stance_predictor_inference():
    from src.inference.predict import StancePredictor
    p = StancePredictor()
    # Test empty input handling
    res_empty = p.predict_stance("")
    assert res_empty["error"] == "empty_input"
    assert res_empty["predicted_label"] is None

    # Test whitespace input
    res_ws = p.predict_stance("   ")
    assert res_ws["error"] == "empty_input"

    # Test valid input inference
    res = p.predict_stance("जलवायु परिवर्तन एक गंभीर चिंता का विषय है।")
    assert res["predicted_label"] in ["FAVOUR", "AGAINST", "NONE"]
    assert "probabilities" in res
    assert res["language"] == "Hindi"
    assert res["script"] == "Devanagari"

    # Test batch prediction
    batch_res = p.predict_batch(["जलवायु परिवर्तन सच है", "সব নাটক"])
    assert len(batch_res) == 2
