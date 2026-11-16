"""Multilingual preprocessing for Hindi / Bengali social-media stance detection.

Design rules (hard constraints from the project brief):
  * Never strip negation, stance-bearing words or emotion-bearing emoji.
  * Never lowercase Indic text (no case) and never lowercase Latin text in the
    default pipeline, because MuRIL/XLM-R subword inventories are case-aware.
  * Never transliterate, never translate, never romanise.
  * Always keep the raw string next to the processed string.

Two variants:
  minimal   : unicode normalisation, whitespace, URL/mention masking, repeat cap
  enhanced  : minimal + zero-width removal + hashtag segmentation hints +
              emoji spaced out so the tokenizer sees them as separate symbols
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
BENGALI = re.compile(r"[\u0980-\u09FF]")
LATIN = re.compile(r"[A-Za-z]")
EMOJI = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "\uFE0F" "]"
)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
REPEAT_RE = re.compile(r"(.)\1{3,}", flags=re.DOTALL)
MULTISPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
NEWLINE_RE = re.compile(r"[\r\n]+")

# Negation cues are listed only so that tests can assert they survive.
NEGATION_CUES = {
    "hi": ["नहीं", "नही", "मत", "बिल्कुल नहीं", "कदापि", "ना"],
    "bn": ["না", "নয়", "নেই", "কখনো", "কিছুই না"],
    "roman": ["nahi", "nahin", "nai", "na", "mat", "ni", "na hoy"],
}


@dataclass
class Preprocessed:
    text_raw: str
    text: str
    script: str
    variant: str
    flags: dict = field(default_factory=dict)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text)


def detect_script(text: str) -> str:
    """Return dominant script label: Devanagari / Bengali / Latin / Mixed / Other."""
    dev = len(DEVANAGARI.findall(text))
    ben = len(BENGALI.findall(text))
    lat = len(LATIN.findall(text))
    total = dev + ben + lat
    if total == 0:
        return "Other"
    present = sum(x > 0 for x in (dev, ben, lat))
    if present > 1 and min(v for v in (dev, ben, lat) if v) / total > 0.10:
        return "Mixed"
    if dev >= ben and dev >= lat:
        return "Devanagari"
    if ben >= dev and ben >= lat:
        return "Bengali"
    return "Latin"


def mask_urls(text: str, token: str = "<URL>") -> str:
    return URL_RE.sub(token, text)


def mask_mentions(text: str, token: str = "<USER>") -> str:
    return MENTION_RE.sub(token, text)


def keep_hashtags(text: str) -> str:
    """Keep the lexical content of a hashtag, drop only the '#' glyph."""
    return HASHTAG_RE.sub(r"\1", text)


def collapse_repeats(text: str, max_repeat: int = 3) -> str:
    """'bahuttttt' -> 'bahuttt' but harmless for Indic (no combining marks hit)."""
    def _fix(m: re.Match) -> str:
        return m.group(1) * max_repeat

    return REPEAT_RE.sub(_fix, text)


def separate_emoji(text: str) -> str:
    return EMOJI.sub(lambda m: f" {m.group(0)} ", text)


def clean_whitespace(text: str) -> str:
    text = NEWLINE_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def preprocess(text: str, variant: str = "minimal", unicode_form: str = "NFC") -> Preprocessed:
    if text is None:
        text = ""
    raw = text
    out = normalize_unicode(str(text), unicode_form)
    out = mask_urls(out)
    out = mask_mentions(out)
    out = keep_hashtags(out)
    out = collapse_repeats(out, 3)
    if variant == "enhanced":
        out = ZERO_WIDTH.sub("", out)
        out = separate_emoji(out)
    out = clean_whitespace(out)
    return Preprocessed(
        text_raw=raw,
        text=out,
        script=detect_script(raw),
        variant=variant,
        flags={
            "has_url": bool(URL_RE.search(raw)),
            "has_mention": bool(MENTION_RE.search(raw)),
            "has_hashtag": "#" in raw,
            "has_emoji": bool(EMOJI.search(raw)),
            "has_negation": any(c in raw.lower() for c in sum(NEGATION_CUES.values(), [])),
            "len_chars": len(raw),
        },
    )


def is_romanized(text: str, threshold: float = 0.6) -> bool:
    """Heuristic Romanised-Indic detector: Latin script + Indic function words."""
    if detect_script(text) not in {"Latin", "Mixed"}:
        return False
    markers = [
        "hai", "hain", "nahi", "kya", "yeh", "woh", "bohot", "bahut", "accha",
        "ki", "ka", "ko", "me", "mein", "aur", "par", "bhi", "kyunki",
        "ache", "kharap", "khub", "kemon", "tumi", "amar", "ami", "kore",
        "hocche", "hobe", "na", "ta", "er", "jonno",
    ]
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return False
    hits = sum(t in markers for t in tokens)
    return hits / len(tokens) >= threshold / 10
