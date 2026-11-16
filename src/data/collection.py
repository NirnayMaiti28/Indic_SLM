"""Track (a): collection harness for the custom Hindi-Bengali stance corpus.

This module never writes invented annotations. It provides:
  * `CollectionPlan`  - per-(language, script, phenomenon) quotas derived from the guideline
  * `validate_rows`   - hard schema + label + provenance validation with actionable errors
  * `dedup_rows`      - exact and near-duplicate removal before annotation
  * `coverage_report` - which quotas are still unfilled (the honest "what's missing")
  * `youtube_candidates` - a thin wrapper around the official YouTube Data API v3
    (no HTML scraping, no CAPTCHA handling; requires the operator's own API key)
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"FAVOUR", "AGAINST", "NONE"}
HUMAN_STATUSES = {"human_verified", "adjudicated"}
REQUIRED = ("id", "text", "language", "script", "label", "annotation_status", "source")
LANGUAGES = {"Hindi", "Bengali"}
SCRIPTS = {"Devanagari", "Bengali", "Latin"}


@dataclass
class CollectionPlan:
    """Quota table from docs/annotation_guideline.md §6."""
    devanagari_hindi: int = 1200
    bengali_bengali: int = 1200
    romanized_each: int = 300
    code_mixed_each: int = 300
    sarcasm_each: int = 150
    negation_each: int = 200
    min_class_share: float = 0.20
    double_annotation_share: float = 0.20

    def targets(self) -> dict[str, dict[str, int]]:
        return {
            "Hindi": {
                "Devanagari": self.devanagari_hindi,
                "Romanized": self.romanized_each,
                "code_mixed": self.code_mixed_each,
                "sarcasm": self.sarcasm_each,
                "negation": self.negation_each,
            },
            "Bengali": {
                "Bengali": self.bengali_bengali,
                "Romanized": self.romanized_each,
                "code_mixed": self.code_mixed_each,
                "sarcasm": self.sarcasm_each,
                "negation": self.negation_each,
            },
        }


@dataclass
class ValidationResult:
    ok: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return len(self.ok)

    def summary(self) -> dict:
        return {
            "accepted": len(self.ok),
            "rejected": len(self.errors),
            "error_kinds": dict(Counter(e["error"] for e in self.errors)),
        }


def validate_rows(rows: Iterable[dict], require_human: bool = True) -> ValidationResult:
    res = ValidationResult()
    seen_ids: set[str] = set()
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED if not str(r.get(k, "")).strip()]
        if missing:
            res.errors.append({"row": i, "error": "missing_fields", "fields": missing})
            continue
        if r["id"] in seen_ids:
            res.errors.append({"row": i, "error": "duplicate_id", "id": r["id"]})
            continue
        if str(r["label"]).upper() not in LABELS:
            res.errors.append({"row": i, "error": "invalid_label", "label": r["label"]})
            continue
        if r["language"] not in LANGUAGES:
            res.errors.append({"row": i, "error": "invalid_language", "language": r["language"]})
            continue
        if r["script"] not in SCRIPTS:
            res.errors.append({"row": i, "error": "invalid_script", "script": r["script"]})
            continue
        status = str(r["annotation_status"]).lower()
        if status == "synthetic" and require_human:
            res.errors.append({"row": i, "error": "synthetic_in_gold_split", "id": r["id"]})
            continue
        if require_human and status not in HUMAN_STATUSES:
            res.errors.append({"row": i, "error": "not_human_verified", "id": r["id"], "status": status})
            continue
        # provenance hygiene: no raw permit/URL, no @handles
        for bad, kind in ((r"https?://", "raw_url_present"), (r"@\w{3,}", "handle_present")):
            if re.search(bad, r["text"]):
                res.errors.append({"row": i, "error": kind, "id": r["id"]})
                break
        else:
            r = dict(r)
            r["label"] = str(r["label"]).upper()
            seen_ids.add(r["id"])
            res.ok.append(r)
    return res


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def dedup_rows(rows: list[dict], near_dup: bool = True, k: int = 5) -> tuple[list[dict], dict]:
    """Exact + shingle-based near-duplicate removal, keeping the first occurrence."""
    seen: dict[str, str] = {}
    kept: list[dict] = []
    exact = 0
    near = 0
    shingle_index: dict[str, str] = {}
    for r in rows:
        key = _norm(r["text"])
        if key in seen:
            exact += 1
            continue
        if near_dup and len(key) > k:
            sh = {key[i : i + k] for i in range(len(key) - k + 1)}
            hit = next((shingle_index[s] for s in sh if s in shingle_index), None)
            if hit is not None:
                near += 1
                seen[key] = hit
                continue
            for s in sh:
                shingle_index.setdefault(s, key)
        seen[key] = key
        kept.append(r)
    return kept, {"input": len(rows), "kept": len(kept), "exact_duplicates_removed": exact,
                  "near_duplicates_removed": near}


def coverage_report(rows: list[dict], plan: CollectionPlan | None = None) -> dict:
    plan = plan or CollectionPlan()
    targets = plan.targets()
    got: dict[str, Counter] = defaultdict(Counter)
    labels: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        lang = r.get("language", "?")
        got[lang][r.get("script", "?")] += 1
        if r.get("romanized"):
            got[lang]["Romanized"] += 1
        if r.get("code_mixed"):
            got[lang]["code_mixed"] += 1
        if r.get("figurative") == "sarcasm":
            got[lang]["sarcasm"] += 1
        if r.get("negation"):
            got[lang]["negation"] += 1
        labels[lang][r.get("label", "?")] += 1

    slots = []
    for lang, need in targets.items():
        for k, target in need.items():
            have = got[lang][k]
            slots.append({"language": lang, "slice": k, "have": have, "target": target,
                          "remaining": max(0, target - have),
                          "pct": round(100 * min(have / target, 1.0), 1)})
    balance = {}
    for lang in targets:
        total = sum(labels[lang].values())
        balance[lang] = {
            lbl: {"n": labels[lang][lbl], "share": round(labels[lang][lbl] / total, 3) if total else 0.0,
                  "below_floor": (labels[lang][lbl] / total < plan.min_class_share) if total else True}
            for lbl in sorted(LABELS)
        }
    total_rows = len(rows)
    return {
        "n_rows": total_rows,
        "quota_slots": slots,
        "remaining_total": sum(s["remaining"] for s in slots),
        "class_balance": balance,
        "labels": {k: dict(v) for k, v in labels.items()},
        "complete": all(s["remaining"] == 0 for s in slots) and total_rows > 0,
    }


def youtube_candidates(query: str, max_results: int = 10) -> list[dict]:
    """Official YouTube Data API v3 search - requires YOUTUBE_API_KEY in env.

    Deliberately does not fall back to HTML scraping: respecting platform terms
    is a hard constraint of this project.
    """
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError(
            "YOUTUBE_API_KEY not set. Provide your own key for the official Data API; "
            "this project does not scrape HTML to bypass platform controls."
        )
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode({
        "part": "snippet", "q": query, "type": "video", "maxResults": max_results,
        "relevanceLanguage": "hi", "key": key,
    })
    url = f"https://www.googleapis.com/youtube/v3/search?{params}"
    with urllib.request.urlopen(url, timeout=30) as fh:
        data = json.load(fh)
    return [
        {"video_id": it["id"]["videoId"], "title": it["snippet"]["title"],
         "channel": it["snippet"]["channelTitle"], "published": it["snippet"]["publishedAt"]}
        for it in data.get("items", [])
    ]


def write_coverage_markdown(rep: dict, path: str | Path) -> Path:
    lines = ["# Custom Corpus — Collection Coverage", "",
             f"Rows ingested: **{rep['n_rows']}** · remaining quota slots: **{rep['remaining_total']}** · complete: **{rep['complete']}**", "",
             "| language | slice | have | target | remaining | % |", "|---|---|---|---|---|---|"]
    for s in rep["quota_slots"]:
        lines.append(f"| {s['language']} | {s['slice']} | {s['have']} | {s['target']} | {s['remaining']} | {s['pct']} |")
    lines += ["", "## Class balance (floor 20 % per language)", "", "| language | FAVOUR | AGAINST | NONE |", "|---|---|---|---|"]
    for lang, d in rep["class_balance"].items():
        cells = []
        for lbl in ("FAVOUR", "AGAINST", "NONE"):
            v = d[lbl]
            cells.append(f"{v['n']} ({v['share']})" + (" ⚠" if v["below_floor"] else ""))
        lines.append(f"| {lang} | " + " | ".join(cells) + " |")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate + dedup + report on a candidate JSONL corpus")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "interim" / "custom_validated.jsonl")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "collection_coverage.md")
    ap.add_argument("--allow-non-human", action="store_true")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    v = validate_rows(rows, require_human=not args.allow_non_human)
    print(json.dumps(v.summary(), indent=2))
    kept, dd = dedup_rows(v.ok)
    print(json.dumps(dd, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept), encoding="utf-8")
    rep = coverage_report(kept)
    write_coverage_markdown(rep, args.report)
    print(json.dumps({"kept": len(kept), "complete": rep["complete"], "remaining": rep["remaining_total"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
