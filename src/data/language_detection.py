"""Language and Script Detection for Indic Social-Media Text.

Supports:
- Devanagari Hindi
- Bengali script
- Romanized Hindi (Hinglish)
- Romanized Bengali (Banglish)
- Code-mixed text (Hindi-English, Bengali-English)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Any


DEVANAGARI_RANGE = (0x0900, 0x097F)
BENGALI_RANGE = (0x0980, 0x09FF)
LATIN_RANGE = (0x0041, 0x007A)

# Typical high-frequency stopwords and markers for Romanized Indic text
HINDI_ROMAN_STOPWORDS = {
    "hai", "hain", "ke", "ki", "ka", "ko", "se", "aur", "bhi", "yeh", "woh",
    "kya", "kyun", "nahi", "nahin", "yaar", "bhai", "par", "pe", "mein", "kar",
    "raha", "rahe", "karna", "hoga", "sach", "bahut", "aaj", "kuch"
}

BENGALI_ROMAN_STOPWORDS = {
    "ache", "nei", "er", "ke", "ar", "o", "kintu", "eta", "ota", "ki", "keno",
    "na", "noy", "dada", "khub", "shob", "korte", "hobe", "ashole", "mone",
    "hoy", "hocche", "bhalo", "ajke", "kothay", "kichu", "bolche"
}


@dataclass
class LanguageProfile:
    text: str
    script: str
    language: str
    script_counts: Dict[str, int]
    is_code_mixed: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "script": self.script,
            "language": self.language,
            "script_counts": self.script_counts,
            "is_code_mixed": self.is_code_mixed,
            "confidence": self.confidence,
        }


def detect_script(text: str) -> str:
    """Identify the dominant script in the given string."""
    counts = {"Devanagari": 0, "Bengali": 0, "Latin": 0, "Other": 0}
    for char in text:
        cp = ord(char)
        if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
            counts["Devanagari"] += 1
        elif BENGALI_RANGE[0] <= cp <= BENGALI_RANGE[1]:
            counts["Bengali"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["Latin"] += 1
        elif not char.isspace() and not unicodedata.category(char).startswith("P"):
            counts["Other"] += 1

    total = sum(counts.values())
    if total == 0:
        return "Unknown"

    dominant = max(counts, key=counts.get)
    if counts[dominant] / total >= 0.60:
        return dominant
    return "Mixed"


def detect_language(text: str) -> str:
    """Detect language (Hindi, Bengali, English, or Unknown)."""
    script = detect_script(text)
    if script == "Devanagari":
        return "Hindi"
    elif script == "Bengali":
        return "Bengali"
    
    # If Latin / Mixed script, check for Romanized Indic words
    tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z]+\b", text)]
    if not tokens:
        return "Unknown"

    hi_hits = sum(1 for t in tokens if t in HINDI_ROMAN_STOPWORDS)
    bn_hits = sum(1 for t in tokens if t in BENGALI_ROMAN_STOPWORDS)

    if hi_hits > bn_hits and hi_hits > 0:
        return "Hindi"
    elif bn_hits > hi_hits and bn_hits > 0:
        return "Bengali"
    elif hi_hits == 0 and bn_hits == 0:
        return "English"
    return "Hindi"  # Default tie-breaker for Romanized Indic


def analyze_language(text: str) -> LanguageProfile:
    """Detailed linguistic profile of the given text."""
    counts = {"Devanagari": 0, "Bengali": 0, "Latin": 0, "Other": 0}
    for char in text:
        cp = ord(char)
        if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
            counts["Devanagari"] += 1
        elif BENGALI_RANGE[0] <= cp <= BENGALI_RANGE[1]:
            counts["Bengali"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["Latin"] += 1
        elif not char.isspace() and not unicodedata.category(char).startswith("P"):
            counts["Other"] += 1

    script = detect_script(text)
    lang = detect_language(text)
    
    total_letters = counts["Devanagari"] + counts["Bengali"] + counts["Latin"]
    is_mixed = False
    if total_letters > 0:
        latin_share = counts["Latin"] / total_letters
        indic_share = (counts["Devanagari"] + counts["Bengali"]) / total_letters
        is_mixed = (latin_share > 0.15 and indic_share > 0.15) or (script == "Mixed")

    confidence = 0.95 if script in ("Devanagari", "Bengali") else 0.80
    return LanguageProfile(
        text=text,
        script=script,
        language=lang,
        script_counts=counts,
        is_code_mixed=is_mixed,
        confidence=confidence
    )
