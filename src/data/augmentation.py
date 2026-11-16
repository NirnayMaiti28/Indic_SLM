"""Stance-preserving augmentation.

Hard rule: a transformation is used only if it cannot flip the stance of a
comment. Every function here is polarity-neutral by construction (it touches
surface form, not propositional content). Even so, all augmented rows are tagged
`synthetic_augmentation` and never counted as human annotation, and the
augmented-vs-original comparison is reported as a separate experiment.
"""
from __future__ import annotations

import random
import re

SPACE_VARIANTS = [" ", "  ", ""]


def char_repeat(text: str, rate: float = 0.15, rng: random.Random | None = None) -> str:
    """Social-media elongation: 'बहुत' -> 'बहुतत', 'nahi' -> 'nahiii'.

    Only the final consonant/vowel of a word is repeated, which is how users
    actually emphasise text and does not alter any morpheme boundary in
    Devanagari/Bengali in a way that changes meaning.
    """
    rng = rng or random.Random(0)
    out = []
    for w in text.split():
        if len(w) > 3 and rng.random() < rate:
            out.append(w + w[-1] * rng.randint(1, 3))
        else:
            out.append(w)
    return " ".join(out)


def spacing_noise(text: str, rate: float = 0.10, rng: random.Random | None = None) -> str:
    """Irregular whitespace, as typed on mobile keyboards."""
    rng = rng or random.Random(0)
    out = []
    for w in text.split():
        sep = rng.choice(SPACE_VARIANTS) if rng.random() < rate else " "
        out.append(w + sep)
    return "".join(out).strip() or text


def emoji_append(text: str, emojis=("🔥", "🌍", "😤", "🙏", "😡", "💔"), rng: random.Random | None = None) -> str:
    """Append a sentiment-bearing emoji. Polarity-neutral on its own: it adds an
    emotional cue without asserting anything about the claim, and it is applied
    with equal probability to every class so it cannot become a label shortcut."""
    rng = rng or random.Random(0)
    return f"{text} {rng.choice(list(emojis))}"


def hashtag_append(text: str, tags=("#ClimateAction", "#जलवायु", "#জলবায়ু", "#SaveEarth"), rng=None) -> str:
    rng = rng or random.Random(0)
    return f"{text} {rng.choice(list(tags))}"


ROMAN_HI = {"है": "hai", "नहीं": "nahi", "और": "aur", "एक": "ek", "यह": "yeh", "गंभीर": "gambhir",
            "समस्या": "samasya", "जलवायु": "jalvayu", "परिवर्तन": "parivartan", "बहुत": "bahut"}
ROMAN_BN = {"হয়": "hoy", "না": "na", "এবং": "ebong", "একটি": "ekti", "গুরুতর": "gurutor",
            "সমস্যা": "samasya", "জলবায়ু": "jolobayu", "পরিবর্তন": "poriborton", "খুব": "khub"}


def partial_romanize(text: str, rate: float = 0.30, rng: random.Random | None = None) -> str:
    """Script-mix by Romanising a subset of *function/lexical* words.

    This simulates the Romanised-Hindi/Bengali slice of the target domain. The
    word's lemma is preserved exactly, so the stance is unchanged; only its
    orthography moves to Latin, which is what real users do.
    """
    rng = rng or random.Random(0)
    table = {**ROMAN_HI, **ROMAN_BN}
    out = []
    for w in text.split():
        core = re.sub(r"[^\w\u0900-\u097F\u0980-\u09FF]", "", w)
        if core in table and rng.random() < rate:
            out.append(table[core] + re.sub(r"[\w\u0900-\u097F\u0980-\u09FF]", "", w))
        else:
            out.append(w)
    return " ".join(out)


def english_codemix(text: str, fillers=("seriously", "actually", "I mean", "bro", "totally"), rng=None) -> str:
    """Insert an English discourse marker without a stance value of its own."""
    rng = rng or random.Random(0)
    words = text.split()
    if len(words) < 2:
        return f"{text} {rng.choice(list(fillers))}"
    pos = rng.randint(1, len(words) - 1)
    words.insert(pos, rng.choice(list(fillers)))
    return " ".join(words)


AUGMENTERS = {
    "char_repeat": char_repeat,
    "spacing": spacing_noise,
    "emoji": emoji_append,
    "hashtag": hashtag_append,
    "romanize": partial_romanize,
    "codemix": english_codemix,
}


def augment(records: list[dict], ops: list[str], copies: int = 1, seed: int = 42,
            keep_original: bool = True) -> list[dict]:
    """Return original + `copies` augmented variants per record, all tagged."""
    rng = random.Random(seed)
    out: list[dict] = []
    if keep_original:
        for r in records:
            out.append({**r, "augmentation": "none"})
    for c in range(copies):
        for r in records:
            text = str(r["text"])
            for op in ops:
                fn = AUGMENTERS[op]
                if op == "char_repeat":
                    text = fn(text, 0.15, rng)
                elif op == "spacing":
                    text = fn(text, 0.10, rng)
                elif op == "romanize":
                    text = fn(text, 0.30, rng)
                elif op == "spacing" or op == "codemix":
                    text = fn(text, rng=rng)
                else:
                    text = fn(text, rng=rng)
            out.append({**r, "id": f"{r.get('id','x')}_aug{c}",
                        "text": text, "augmentation": "+".join(ops),
                        "synthetic_augmentation": True})
    return out


def verify_label_support(records: list[dict]) -> dict:
    """Sanity check: augmentation must not change the class distribution shape."""
    from collections import Counter

    return dict(Counter(str(r.get("label")) for r in records))
