"""Unified dataset loading for the supervisable corpora.

`load_gwstance` returns English development data with **derived** labels
(posterior argmax). It is the only corpus in this project with a label space
compatible with the target claim, and it is never presented as Hindi/Bengali
supervision. `load_custom` reads the (future) human-annotated corpus and refuses
to return rows whose `annotation_status` is not human-derived.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

LABELS = ["FAVOUR", "AGAINST", "NONE"]
POSTERIOR_MAP = {"agree": "FAVOUR", "disagree": "AGAINST", "neutral": "NONE"}
VOTE_MAP = {"agrees": "FAVOUR", "disagrees": "AGAINST", "neutral": "NONE"}

HUMAN_STATUSES = {"human_verified", "adjudicated"}


def load_gwstance(drop_screening: bool = True) -> list[dict]:
    """GWSD -> project label space via Bayesian posterior argmax.

    The first five rows are the annotators' screening sentences and carry no
    posterior (documented upstream); they are dropped by default.
    """
    path = RAW / "gwstance" / "GWSD.tsv"
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for i, r in enumerate(reader):
            post: dict[str, float] = {}
            for k in ("agree", "disagree", "neutral"):
                v = (r.get(k) or "").strip()
                if v not in ("", "NA", "nan", "None"):
                    try:
                        post[k] = float(v)
                    except ValueError:
                        pass
            if post:
                label = POSTERIOR_MAP[max(post, key=post.get)]
                confidence = max(post.values())
            else:
                if drop_screening:
                    continue
                votes = Counter((r.get(f"worker_{w}") or "").strip() for w in range(8))
                top = votes.most_common(1)[0][0]
                if top not in VOTE_MAP:
                    continue
                label = VOTE_MAP[top]
                confidence = votes[top] / 8

            rows.append({
                "id": f"gwsd_{i:05d}",
                "text": (r.get("sentence") or "").strip(),
                "label": label,
                "label_source": "derived_posterior_argmax" if post else "derived_worker_majority",
                "label_confidence": round(float(confidence), 4),
                "language": "English",
                "script": "Latin",
                "source": "GWSD",
                "guid": r.get("guid"),
                "in_held_out_test": (r.get("in_held_out_test") or "").strip().lower() == "true",
            })
    return [r for r in rows if r["text"]]


def load_custom(path: str | Path | None = None, include_non_human: bool = False) -> list[dict]:
    """Load the custom Hindi/Bengali corpus (JSONL).

    Rows must carry `annotation_status`. Non-human rows (synthetic, pipeline
    tests, pending) are excluded unless explicitly requested, and even then they
    are tagged so no downstream step can mistake them for gold.
    """
    p = Path(path) if path else RAW / "custom_climate_dataset" / "corpus.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        status = str(r.get("annotation_status", "")).lower()
        is_human = status in HUMAN_STATUSES
        if not is_human and not include_non_human:
            continue
        r["_is_human"] = is_human
        rows.append(r)
    return rows


def dataset_stats(records: Iterable[dict]) -> dict:
    recs = list(records)
    return {
        "n": len(recs),
        "labels": dict(Counter(r["label"] for r in recs)),
        "languages": dict(Counter(str(r.get("language", "")) for r in recs)),
        "scripts": dict(Counter(str(r.get("script", "")) for r in recs)),
        "label_sources": dict(Counter(str(r.get("label_source", "")) for r in recs)),
    }


def write_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
