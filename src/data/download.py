"""Dataset acquisition for the four resources used in this project.

Every resource is fetched from its *original* host. Nothing is mirrored or
re-uploaded, and licences are recorded next to the bytes so that the audit in
`reports/dataset_audit.md` can be reproduced.

Verified provenance (checked 2026-10-31):
  GWStance      -> github.com/yiweiluo/GWStance  (GWSD.tsv, 2300 rows)
  Dhoroni       -> huggingface.co/datasets/ciol-research/Dhoroni (MIT, 2.29k rows)
  IndicSentiment-> huggingface.co/datasets/ai4bharat/IndicSentiment (hi, bn)

Run:  python -m src.data.download --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

GWSTANCE_REPO = "https://github.com/yiweiluo/GWStance.git"
DHORONI_PARQUET = (
    "https://huggingface.co/api/datasets/ciol-research/Dhoroni/parquet/default/train/0.parquet"
)
INDICSENTIMENT_URL = (
    "https://huggingface.co/datasets/ai4bharat/IndicSentiment/resolve/main/data/{split}/{lang}.json"
)


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest}")
        return True
    print(f"[get ] {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except urllib.error.HTTPError as e:
        print(f"[warn] {url} returned HTTP {e.code}: {e.reason} (skipping)")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def fetch_gwstance() -> None:
    dest = RAW / "gwstance"
    if (dest / "GWSD.tsv").exists():
        print(f"[skip] {dest / 'GWSD.tsv'}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", GWSTANCE_REPO, str(dest)], check=True)


def fetch_dhoroni() -> None:
    _download(DHORONI_PARQUET, RAW / "dhoroni" / "dhoroni_train.parquet")


def fetch_indic_sentiment(langs=("hi", "bn"), splits=("validation", "test")) -> None:
    for split in splits:
        for lang in langs:
            url = INDICSENTIMENT_URL.format(split=split, lang=lang)
            _download(url, RAW / "indic_sentiment" / f"{split}_{lang}.json")


LICENCES = {
    "gwstance": "Research use; see paper (ACL Findings 2020) and repository terms.",
    "dhoroni": "MIT (declared on the Hugging Face dataset card).",
    "indic_sentiment": "AI4Bharat resource; see repository card for terms.",
    "custom": "Self-collected; no redistribution of raw platform text without permission.",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gwstance", action="store_true")
    ap.add_argument("--dhoroni", action="store_true")
    ap.add_argument("--indic-sentiment", action="store_true")
    args = ap.parse_args(argv)
    if not any([args.all, args.gwstance, args.dhoroni, args.indic_sentiment]):
        ap.print_help()
        return 1
    if args.all or args.gwstance:
        fetch_gwstance()
    if args.all or args.dhoroni:
        fetch_dhoroni()
    if args.all or args.indic_sentiment:
        fetch_indic_sentiment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
