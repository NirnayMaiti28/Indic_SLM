"""Inter-annotator agreement (IAA) for the custom stance corpus.

Implements Cohen's kappa (two annotators, complete design) and Krippendorff's
alpha (nominal, any number of annotators, missing values allowed).

Usage:
    python -m src.data.agreement --csv data/raw/custom_climate_dataset/annotations_A.csv \
        --csv-b data/raw/custom_climate_dataset/annotations_B.csv
    python -m src.data.agreement --wide data/raw/custom_climate_dataset/iaa_wide.csv

Expected wide format:  id,annotator_1,annotator_2[,annotator_3...]
Long format (two files): id,label
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def cohens_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a, "need equal, non-empty rating lists"
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    labels = set(ca) | set(cb)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha_nominal(units: list[list[str]]) -> float:
    """units = list of per-item rating lists (missing values simply absent)."""
    pairable = [u for u in units if len(u) > 1]
    if not pairable:
        return float("nan")
    # observed disagreement
    do_num = 0.0
    n_total = 0
    for u in pairable:
        m = len(u)
        counts = Counter(u)
        do_num += (m * m - sum(c * c for c in counts.values())) / (m - 1)
        n_total += m
    do = do_num / n_total
    # expected disagreement
    all_vals = [v for u in pairable for v in u]
    N = len(all_vals)
    counts = Counter(all_vals)
    de = (N * N - sum(c * c for c in counts.values())) / (N * (N - 1))
    if de == 0:
        return 1.0
    return 1.0 - do / de


def _read_pairs(csv_a: Path, csv_b: Path) -> tuple[list[str], list[str]]:
    def load(p: Path) -> dict[str, str]:
        with p.open(encoding="utf-8") as fh:
            return {r["id"]: r["label"] for r in csv.DictReader(fh)}

    A, B = load(csv_a), load(csv_b)
    ids = sorted(set(A) & set(B))
    return [A[i] for i in ids], [B[i] for i in ids]


def _read_wide(p: Path) -> list[list[str]]:
    units = []
    with p.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = [c for c in reader.fieldnames or [] if c != "id"]
        for row in reader:
            vals = [row[c].strip() for c in cols if row.get(c, "").strip()]
            if vals:
                units.append(vals)
    return units


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, help="annotator A file (id,label)")
    ap.add_argument("--csv-b", type=Path, help="annotator B file (id,label)")
    ap.add_argument("--wide", type=Path, help="single wide file (id,annotator_1,...)")
    args = ap.parse_args(argv)

    if args.wide:
        units = _read_wide(args.wide)
        print(f"units: {len(units)}")
        print(f"Krippendorff alpha (nominal): {krippendorff_alpha_nominal(units):.4f}")
        if all(len(u) == 2 for u in units):
            print(f"Cohen kappa: {cohens_kappa([u[0] for u in units], [u[1] for u in units]):.4f}")
        return 0

    if args.csv and args.csv_b:
        a, b = _read_pairs(args.csv, args.csv_b)
        print(f"paired items: {len(a)}")
        print(f"Cohen kappa: {cohens_kappa(a, b):.4f}")
        print(f"Krippendorff alpha (nominal): {krippendorff_alpha_nominal([[x, y] for x, y in zip(a, b)]):.4f}")
        conf = Counter(zip(a, b))
        print("confusion (A,B):", dict(conf))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
