"""Phase 8 CLI - screen teacher-candidate tokenizers on real Hindi/Bengali text.

    python -m src.training.screen_teachers
    python -m src.training.screen_teachers --max-examples 300

Corpora used (all real bytes, no synthetic substitution):
  Hindi   : IndicSentiment `INDIC REVIEW` (Devanagari) - the repo's Indic probe
  Bengali : IndicSentiment `INDIC REVIEW` (Bengali script)
  English : GWStance/GWSD sentences - shown only for contrast, since the target
            task is Hindi+Bengali and an English number cannot justify a choice.

Every candidate is attempted with AutoTokenizer. A candidate that cannot be
loaded (gated repo, missing tokenizer files, network failure) is recorded with
its error string and marked `status: not_screened` - it is never silently dropped
and never given an invented score.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.tokenizer_screening import FragmentationReport, rank_candidates, screen_tokenizer  # noqa: E402

# Candidate pool: multilingual encoders with claimed Hindi+Bengali coverage, plus
# two monolingual Indic encoders as a tokenizer-efficiency upper bound.
CANDIDATES = [
    ("google/muril-base-cased", "multilingual (17 Indic + transliterated)"),
    ("google/muril-large-cased", "multilingual large"),
    ("ai4bharat/IndicBERTv2-MLM-only", "multilingual Indic (IndicCorp v2)"),
    ("ai4bharat/indic-bert", "multilingual Indic v1 (ALBERT)"),
    ("xlm-roberta-base", "multilingual 100-lang"),
    ("xlm-roberta-large", "multilingual large"),
    ("bert-base-multilingual-cased", "mBERT"),
    ("distilbert-base-multilingual-cased", "distilled mBERT"),
    ("l3cube-pune/hindi-bert-v2", "monolingual Hindi"),
    ("l3cube-pune/bengali-bert", "monolingual Bengali"),
]


def load_probe(max_examples: int | None = None) -> dict[str, list[str]]:
    corpora: dict[str, list[str]] = {"Hindi": [], "Bengali": [], "English": []}
    for lang, key in (("hi", "Hindi"), ("bn", "Bengali")):
        for split in ("validation", "test"):
            f = ROOT / "data" / "raw" / "indic_sentiment" / f"{split}_{lang}.json"
            if not f.exists() or f.stat().st_size < 100:
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                txt = (json.loads(line).get("INDIC REVIEW") or "").strip()
                if txt:
                    corpora[key].append(txt)

    gw = ROOT / "data" / "raw" / "gwstance" / "GWSD.tsv"
    if gw.exists():
        import csv

        with gw.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                s = (r.get("sentence") or "").strip()
                if s:
                    corpora["English"].append(s)

    if max_examples:
        corpora = {k: v[:max_examples] for k, v in corpora.items()}
    return corpora


def screen_all(max_examples: int | None = None) -> dict:
    from transformers import AutoTokenizer

    corpora = load_probe(max_examples)
    print("corpus sizes: " + ", ".join(f"{k}={len(v)}" for k, v in corpora.items()))

    rows: list[FragmentationReport] = []
    failures: list[dict] = []
    meta: list[dict] = []

    for name, family in CANDIDATES:
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # gated / missing / network
            failures.append({"model": name, "family": family, "status": "not_screened",
                             "error": f"{type(exc).__name__}: {str(exc)[:180]}"})
            print(f"[FAIL] {name:44s} {type(exc).__name__}")
            continue

        meta.append({"model": name, "family": family, "status": "screened",
                     "tokenizer_class": type(tok).__name__,
                     "vocab_size": int(getattr(tok, "vocab_size", 0) or 0),
                     "is_fast": bool(getattr(tok, "is_fast", False)),
                     "load_seconds": round(time.time() - t0, 2)})

        for language, texts in corpora.items():
            if not texts:
                continue
            rep = screen_tokenizer(name, tok, texts, language)
            rows.append(rep)
            print(f"[ OK ] {name:44s} {language:8s} fert={rep.fertility:.3f} "
                  f"cpt={rep.chars_per_token:.2f} unk={rep.unk_rate:.4f} "
                  f"single={rep.single_char_share:.3f} trunc@64={rep.truncation_rate_at_64:.3f}")

    ranked = rank_candidates(rows)
    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_sizes": {k: len(v) for k, v in corpora.items()},
        "max_length_budget": 64,
        "ranking_criterion": "mean fertility over Hindi+Bengali, ascending; ties -> chars/token desc, truncation asc",
        "candidates": meta,
        "failures": failures,
        "per_language": [r.to_dict() for r in rows],
        "ranking": ranked,
    }
    res = ROOT / "experiments" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "tokenizer_screening.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(ranked, res / "tokenizer_screening.csv")
    write_report(out)
    return out


def write_csv(ranked: list[dict], path: Path) -> None:
    import csv

    if not ranked:
        return
    cols = ["rank", "model", "fertility_mean", "fertility_hi", "fertility_bn", "fertility_en",
            "chars_per_token_mean", "single_char_share_mean", "unk_rate_mean", "trunc_rate_mean",
            "mean_tokens_hi", "mean_tokens_bn", "p95_tokens_hi", "p95_tokens_bn", "vocab_size", "tokenizer_class"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in ranked:
            w.writerow(r)


def write_report(out: dict) -> None:
    cs = out["corpus_sizes"]
    lines = [
        "# Teacher Candidate Screening — Phase 8", "",
        f"Generated {out['generated_utc']} by `python -m src.training.screen_teachers`.", "",
        "## Method", "",
        "Tokenizers only — no weights are loaded, so this screening costs no GPU and cannot leak evaluation data.",
        "Metrics are computed from real tokenizer runs over real Indic bytes:", "",
        "| metric | definition | direction |", "|---|---|---|",
        "| fertility | subword tokens / whitespace word | lower is better |",
        "| chars/token | characters consumed per subword token | higher is better |",
        "| UNK rate | share of unknown tokens | lower is better |",
        "| single-char share | share of 1-character tokens (word must be rebuilt from graphemes) | lower is better |",
        f"| truncation @ {out['max_length_budget']} | share of examples exceeding the project's CPU token budget | lower is better |",
        "", "**Ranking criterion:** " + out["ranking_criterion"] + ".", "",
        "## Corpora", "",
        f"- Hindi: {cs.get('Hindi', 0)} IndicSentiment `INDIC REVIEW` sentences (Devanagari script)",
        f"- Bengali: {cs.get('Bengali', 0)} IndicSentiment `INDIC REVIEW` sentences (Bengali script)",
        f"- English: {cs.get('English', 0)} GWSD sentences — contrast only, **not** used for ranking", "",
        "IndicSentiment is translated product reviews: it is a valid *tokenizer-coverage* probe because",
        "the bytes are genuine Devanagari and Bengali, but it is not climate-domain text and is never",
        "used here to claim stance performance.", "",
        "## Ranked results", "",
        "| # | model | fertility hi | fertility bn | mean | chars/tok | single-char | UNK | trunc@64 | p95 hi / bn | vocab |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out["ranking"]:
        lines.append(
            f"| {r['rank']} | `{r['model']}` | {r['fertility_hi']:.3f} | {r['fertility_bn']:.3f} | "
            f"**{r['fertility_mean']:.3f}** | {r['chars_per_token_mean']:.2f} | {r['single_char_share_mean']:.3f} | "
            f"{r['unk_rate_mean']:.4f} | {r['trunc_rate_mean']:.3f} | {r['p95_tokens_hi']} / {r['p95_tokens_bn']} | "
            f"{r['vocab_size']:,} |"
        )
    if out["failures"]:
        lines += ["", "## Not screened", "",
                  "| model | status | error |", "|---|---|---|"]
        for f in out["failures"]:
            lines.append(f"| `{f['model']}` | {f['status']} | `{f['error'][:120]}` |")
    lines += ["", "## Selection", "",
              f"Rank 1 is `{out['ranking'][0]['model']}` at mean fertility {out['ranking'][0]['fertility_mean']:.3f} "
              f"(Hindi {out['ranking'][0]['fertility_hi']:.3f}, Bengali {out['ranking'][0]['fertility_bn']:.3f}). "
              "Selection is not automatic: the top-ranked tokenizer is adopted as **teacher** only if it is also "
              "trainable inside the measured CPU budget and licensed for this use, which is checked before any fine-tuning.", ""]
    p = ROOT / "reports" / "tokenizer_screening.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args(argv)
    out = screen_all(args.max_examples)
    print("\n=== RANKING ===")
    for r in out["ranking"]:
        print(f"{r['rank']:2d}. {r['model']:44s} mean={r['fertility_mean']:.3f} "
              f"hi={r['fertility_hi']:.3f} bn={r['fertility_bn']:.3f}")
    if out["failures"]:
        print("\n=== NOT SCREENED ===")
        for f in out["failures"]:
            print(f"- {f['model']}: {f['error'][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
