"""Group-aware, leakage-controlled splitting.

Why this module exists: the Phase-3 audit found **253 exact duplicate
sentences** in GWSD (2 300 rows, 11 %). A naive random split therefore leaks
roughly a quarter of the test set into training. Every splitter here groups by a
normalised text key so that no near-identical item can straddle two splits.

Strategy ladder (all stratified on the label):
  random_stratified   : reference only, leaks duplicates -> reported as a control
  group_stratified    : default; groups = normalised text, stratified on label
  source_held_out     : hold out entire source/group families (GWSD `guid` round)
  language_held_out   : leave-one-language-out for the custom corpus (hi -> bn, bn -> hi)
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def text_key(text: str) -> str:
    """Normalised grouping key: kills case, punctuation runs and whitespace."""
    t = re.sub(r"\s+", " ", str(text).strip().lower())
    t = re.sub(r"[^\w\u0900-\u097F\u0980-\u09FF ]+", "", t)
    return t


def near_dup_buckets(texts: Sequence[str], k: int = 5) -> list[list[int]]:
    """Union-find over shared character k-shingles (fast, no pair enumeration)."""
    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    index: dict[str, int] = {}
    for i, t in enumerate(texts):
        norm = re.sub(r"\s+", " ", str(t).strip().lower())
        shingles = {norm[j : j + k] for j in range(max(0, len(norm) - k + 1))} or {norm}
        for s in shingles:
            if s in index:
                union(index[s], i)
            else:
                index[s] = i

    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(len(texts)):
        buckets[find(i)].append(i)
    return list(buckets.values())


@dataclass
class SplitReport:
    strategy: str
    n_train: int
    n_val: int
    n_test: int
    n_groups: int
    train_test_group_overlap: int
    label_distribution: dict
    leakage_safe: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _stratified_assign(
    labels: Sequence[str], groups: dict[str, list[int]], fractions: tuple[float, float, float], rng: np.random.Generator
) -> dict[str, list[int]]:
    """Assign whole groups to train/val/test, stratifying on the group's majority label."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for g, members in groups.items():
        majority = Counter(labels[i] for i in members).most_common(1)[0][0]
        by_label[majority].append(g)

    out = {"train": [], "val": [], "test": []}
    f_tr, f_va, _ = fractions
    for lbl, glist in by_label.items():
        rng.shuffle(glist)
        n = len(glist)
        n_tr = int(round(n * f_tr))
        n_va = int(round(n * f_va))
        for j, g in enumerate(glist):
            if j < n_tr:
                out["train"].extend(groups[g])
            elif j < n_tr + n_va:
                out["val"].extend(groups[g])
            else:
                out["test"].extend(groups[g])
    return out


def group_stratified_split(
    records: Sequence[dict],
    label_key: str = "label",
    text_key_name: str = "text",
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    use_near_dup: bool = False,
) -> tuple[dict[str, list[dict]], SplitReport]:
    rng = np.random.default_rng(seed)
    texts = [r[text_key_name] for r in records]
    labels = [str(r[label_key]) for r in records]

    if use_near_dup:
        buckets = near_dup_buckets(texts)
        groups = {f"b{i}": members for i, members in enumerate(buckets)}
    else:
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[text_key(r[text_key_name])].append(i)
        groups = dict(groups)

    idx = _stratified_assign(labels, groups, fractions, rng)
    split = {k: [records[i] for i in sorted(v)] for k, v in idx.items()}

    train_keys = {text_key(r[text_key_name]) for r in split["train"]}
    test_keys = {text_key(r[text_key_name]) for r in split["test"]}
    overlap = len(train_keys & test_keys)

    rep = SplitReport(
        strategy="group_stratified" + ("+near_dup" if use_near_dup else ""),
        n_train=len(split["train"]),
        n_val=len(split["val"]),
        n_test=len(split["test"]),
        n_groups=len(groups),
        train_test_group_overlap=overlap,
        label_distribution={
            s: dict(Counter(str(r[label_key]) for r in split[s])) for s in ("train", "val", "test")
        },
        leakage_safe=overlap == 0,
    )
    return split, rep


def random_stratified_split(
    records: Sequence[dict],
    label_key: str = "label",
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[dict[str, list[dict]], SplitReport]:
    """Deliberately leaky control: kept so the leakage effect can be quantified."""
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_label[str(r[label_key])].append(i)

    idx = {"train": [], "val": [], "test": []}
    f_tr, f_va, _ = fractions
    for lbl, ids in by_label.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(round(n * f_tr)), int(round(n * f_va))
        idx["train"].extend(ids[:n_tr])
        idx["val"].extend(ids[n_tr : n_tr + n_va])
        idx["test"].extend(ids[n_tr + n_va :])

    split = {k: [records[i] for i in sorted(v)] for k, v in idx.items()}
    tk = {text_key(r["text"]) for r in split["train"]}
    te = {text_key(r["text"]) for r in split["test"]}
    rep = SplitReport(
        strategy="random_stratified",
        n_train=len(split["train"]),
        n_val=len(split["val"]),
        n_test=len(split["test"]),
        n_groups=len(records),
        train_test_group_overlap=len(tk & te),
        label_distribution={s: dict(Counter(str(r[label_key]) for r in split[s])) for s in ("train", "val", "test")},
        leakage_safe=len(tk & te) == 0,
    )
    return split, rep


def language_held_out_split(records: Sequence[dict], held_out_lang: str, label_key: str = "label") -> tuple[dict[str, list[dict]], dict]:
    """Zero-shot transfer protocol for the custom bilingual corpus."""
    test = [r for r in records if str(r.get("language", "")).lower() == held_out_lang.lower()]
    rest = [r for r in records if str(r.get("language", "")).lower() != held_out_lang.lower()]
    train, rep = group_stratified_split(rest, label_key=label_key, fractions=(0.85, 0.15, 0.0), seed=42)
    out = {"train": train["train"], "val": train["val"], "test": test}
    info = {
        "protocol": "language_held_out",
        "held_out_language": held_out_lang,
        "n_train": len(out["train"]),
        "n_val": len(out["val"]),
        "n_test": len(out["test"]),
        "reported_f1": "macro_f1 on held-out language only",
    }
    return out, info


def save_splits(split: dict[str, list[dict]], report: SplitReport | dict, out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "report": report.to_dict() if isinstance(report, SplitReport) else report,
        "splits": {k: v for k, v in split.items()},
    }
    p = out / f"{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(records: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(str(r.get("text", "")).encode("utf-8"))
    return h.hexdigest()[:16]
