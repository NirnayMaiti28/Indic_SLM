import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (  # noqa: E402
    NEGATION_CUES,
    collapse_repeats,
    detect_script,
    is_romanized,
    preprocess,
)


def test_devanagari_preserved():
    out = preprocess("जलवायु परिवर्तन एक गंभीर समस्या है")
    assert out.script == "Devanagari"
    assert "जलवायु" in out.text


def test_bengali_preserved():
    out = preprocess("জলবায়ু পরিবর্তন একটি গুরুতর সমস্যা")
    assert out.script == "Bengali"
    assert "জলবায়ু" in out.text


def test_negation_survives_both_variants():
    for variant in ("minimal", "enhanced"):
        hi = preprocess("ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है", variant=variant)
        bn = preprocess("জলবায়ু পরিবর্তন কোনো গুরুতর সমস্যা নয়", variant=variant)
        assert "नहीं" in hi.text and hi.flags["has_negation"]
        assert "নয়" in bn.text and bn.flags["has_negation"]


def test_url_and_mention_masked():
    out = preprocess("dekhо https://t.co/x @rahul yeh sahi hai")
    assert "<URL>" in out.text and "<USER>" in out.text


def test_emoji_kept():
    out = preprocess("पर्यावरण बचाओ 🌍🔥", variant="enhanced")
    assert "🌍" in out.text


def test_hashtag_text_kept():
    out = preprocess("#ClimateAction जरूरी है")
    assert "ClimateAction" in out.text


def test_repeat_collapse():
    assert collapse_repeats("bahuttttttt") == "bahuttt"
    assert collapse_repeats("नहीं।।।।।") == "नहीं।।।"


def test_no_lowercasing_of_latin():
    out = preprocess("Climate Change IS Real")
    assert "Climate" in out.text and "IS" in out.text


def test_script_detection_mixed():
    assert detect_script("climate परिवर्तन serious है") == "Mixed"
    assert detect_script("jolobayu poriborton") == "Latin"


def test_romanized_detection():
    assert is_romanized("yeh baat sahi hai bhai")
    assert not is_romanized("जलवायु परिवर्तन")


def test_empty_input():
    out = preprocess("")
    assert out.text == "" and out.script == "Other"


def test_none_input():
    out = preprocess(None)  # type: ignore[arg-type]
    assert out.text == ""


def test_negation_cue_table_nonempty():
    assert all(NEGATION_CUES[k] for k in NEGATION_CUES)
