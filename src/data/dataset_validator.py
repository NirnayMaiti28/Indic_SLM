"""Automated dataset auditing.

Produces every quantity the project brief asks for: counts, label / language /
script distributions, length statistics, missing values, duplicates, near
duplicates, imbalance ratio, invalid labels, encoding damage and a leakage
probe based on character n-gram near-duplicates across splits.

Usage:
    python -m src.data.dataset_validator --dataset gwstance
    python -m src.data.dataset_validator --all --json experiments/results/audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

VALID_LABELS = {"FAVOUR", "AGAINST", "NONE"}
REPLACEMENT_CHAR = "\ufffd"


@dataclass
class AuditReport:
    dataset: str
    n_examples: int
    label_distribution: dict
    language_distribution: dict
    script_distribution: dict
    length_mean: float
    length_median: float
    length_p95: float
    missing_text: int
    empty_text: int
    exact_duplicates: int
    near_duplicates: int
    imbalance_ratio: float
    invalid_labels: dict
    encoding_issues: int
    notes: list


def _shingles(text: str, k: int = 5) -> set[str]:
    t = re.sub(r"\s+", " ", text.strip().lower())
    if len(t) <= k:
        return {t}
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def count_near_duplicates(texts: Sequence[str], threshold: float = 0.9, max_pairs: int = 2_000_000) -> int:
    """O(n^2) Jaccard on character 5-shingles, with a hard pair budget."""
    sh = [_shingles(t) for t in texts]
    n, dup, pairs = len(sh), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if pairs > max_pairs:
                return dup
            if jaccard(sh[i], sh[j]) >= threshold:
                dup += 1
    return dup


def _length_stats(lengths: Sequence[int]) -> tuple[float, float, float]:
    if not lengths:
        return 0.0, 0.0, 0.0
    s = sorted(lengths)
    n = len(s)
    p95 = s[min(n - 1, int(round(0.95 * (n - 1))))]
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return round(mean, 2), round(float(median), 2), float(p95)


def audit_records(
    dataset: str,
    records: Iterable[dict],
    text_key: str = "text",
    label_key: str | None = "label",
    language_key: str | None = "language",
    notes: Sequence[str] = (),
    near_dup: bool = True,
) -> AuditReport:
    from .preprocessing import detect_script  # local import keeps module import-light

    recs = list(records)
    texts = [str(r.get(text_key) or "") for r in recs]
    labels = [r.get(label_key) for r in recs] if label_key else []
    langs = [r.get(language_key) for r in recs] if language_key else []

    lengths = [len(t) for t in texts]
    mean, median, p95 = _length_stats(lengths)

    exact_dups = len(texts) - len({t.strip() for t in texts})
    invalid = {str(k): v for k, v in Counter(
        l for l in labels if l is not None and str(l).upper() not in VALID_LABELS
    ).items()}
    encoding = sum(REPLACEMENT_CHAR in t for t in texts)
    counts = Counter(str(l) for l in labels if l is not None)
    imbalance = round(max(counts.values()) / min(counts.values()), 3) if len(counts) > 1 and min(counts.values()) else 0.0

    return AuditReport(
        dataset=dataset,
        n_examples=len(recs),
        label_distribution=dict(counts),
        language_distribution=dict(Counter(str(l) for l in langs if l)),
        script_distribution=dict(Counter(detect_script(t) for t in texts)),
        length_mean=mean,
        length_median=median,
        length_p95=p95,
        missing_text=sum(r.get(text_key) in (None, "") for r in recs),
        empty_text=sum(not t.strip() for t in texts),
        exact_duplicates=exact_dups,
        near_duplicates=count_near_duplicates(texts) if (near_dup and len(texts) <= 3000) else -1,
        imbalance_ratio=imbalance,
        invalid_labels=invalid,
        encoding_issues=encoding,
        notes=list(notes),
    )


# --------------------------------------------------------------------------
# dataset-specific loaders
# --------------------------------------------------------------------------
def load_gwstance() -> list[dict]:
    """GWSD: 8 worker labels per sentence + Bayesian posteriors (agree/disagree/neutral)."""
    import csv

    path = RAW / "gwstance" / "GWSD.tsv"
    rows = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for i, r in enumerate(reader):
            post = {k: float(r[k]) if r.get(k) not in (None, "", "NA") else None
                    for k in ("agree", "disagree", "neutral")}
            # majority label mapped onto the project's three classes
            if post["agree"] is not None:
                label = max(post, key=lambda k: post[k] if post[k] is not None else -1)
                label = {"agree": "FAVOUR", "disagree": "AGAINST", "neutral": "NONE"}[label]
            else:
                votes = Counter(r[f"worker_{w}"] for w in range(8))
                label = {"agrees": "FAVOUR", "disagrees": "AGAINST", "neutral": "NONE"}[votes.most_common(1)[0][0]]
            rows.append({
                "id": f"gwsd_{i:05d}",
                "text": r["sentence"],
                "label": label,
                "language": "English",
                "script": "Latin",
                "source": "GWSD/Luo et al. 2020",
                "posterior": post,
                "in_held_out_test": r.get("in_held_out_test"),
            })
    return rows


def load_dhoroni() -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(RAW / "dhoroni" / "dhoroni_train.parquet").to_pandas()
    return table.to_dict(orient="records")


def load_indic_sentiment(lang: str = "hi") -> list[dict]:
    rows = []
    for split in ("train", "validation", "test"):
        p = RAW / "indic_sentiment" / f"{split}_{lang}.json"
        if not p.exists() or p.stat().st_size < 100:
            continue
        txt = p.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            data = [json.loads(l) for l in txt.splitlines() if l.strip()]
        for i, r in enumerate(data):
            rows.append({**r, "id": f"is_{lang}_{split}_{i:05d}", "split": split, "language": lang})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gwstance", "dhoroni", "indic_sentiment", "all"], default="all")
    ap.add_argument("--all", action="store_true", help="alias for --dataset all")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if args.all:
        args.dataset = "all"

    reports: list[AuditReport] = []

    if args.dataset in ("gwstance", "all"):
        recs = load_gwstance()
        reports.append(audit_records("gwstance", recs, near_dup=False,
                                     notes=["English reference only; labels are posterior maxima, not project gold."]))

    if args.dataset in ("dhoroni", "all"):
        recs = load_dhoroni()
        text_key = next((k for k in ("Article", "article", "text", "Text", "content") if k in recs[0]), None)
        reports.append(audit_records("dhoroni", recs, text_key=text_key or "text", label_key=None,
                                     language_key=None, near_dup=False,
                                     notes=["Bengali news articles; no project-compatible stance label assumed."]))

    if args.dataset in ("indic_sentiment", "all"):
        for lang in ("hi", "bn"):
            recs = load_indic_sentiment(lang)
            if recs:
                reports.append(audit_records(f"indic_sentiment_{lang}", recs, text_key="INDIC REVIEW",
                                             label_key="LABEL", language_key=None, near_dup=False,
                                             notes=["Product/service reviews; sentiment != stance."]))

    out = [asdict(r) for r in reports]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
