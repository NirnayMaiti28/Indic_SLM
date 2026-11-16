import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.splitting import (  # noqa: E402
    group_stratified_split, language_held_out_split, random_stratified_split, text_key,
)
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_macro_f1, compute_metrics, mcnemar_test, per_language_report,
)

LABELS = ["FAVOUR", "AGAINST", "NONE"]


def _mk(n_per=40, lang="English"):
    recs = []
    for i in range(n_per):
        for lbl in LABELS:
            recs.append({"text": f"{lbl} sentence number {i} about climate", "label": lbl, "language": lang})
    return recs


def test_group_split_has_no_text_overlap():
    recs = _mk()
    # inject duplicates deliberately
    recs += [dict(r) for r in recs[:10]]
    split, rep = group_stratified_split(recs, seed=42)
    assert rep.leakage_safe is True
    assert rep.train_test_group_overlap == 0
    tr = {text_key(r["text"]) for r in split["train"]}
    te = {text_key(r["text"]) for r in split["test"]}
    assert not (tr & te)


def test_random_split_leaks_duplicates():
    recs = _mk()
    recs += [dict(r) for r in recs[:20]]
    _, rep = random_stratified_split(recs, seed=42)
    assert rep.leakage_safe is False
    assert rep.train_test_group_overlap > 0


def test_split_sizes_roughly_respected():
    recs = _mk()
    split, rep = group_stratified_split(recs, seed=7)
    total = sum(len(v) for v in split.values())
    assert total == len(recs)
    assert 0.6 <= rep.n_train / total <= 0.8


def test_split_is_stratified():
    recs = _mk()
    _, rep = group_stratified_split(recs, seed=42)
    for s in ("train", "val", "test"):
        d = rep.label_distribution[s]
        assert set(d) <= set(LABELS)
        assert all(v > 0 for v in d.values())


def test_language_held_out():
    recs = [{"text": f"hindi {i}", "label": LABELS[i % 3], "language": "Hindi"} for i in range(30)]
    recs += [{"text": f"bangla {i}", "label": LABELS[i % 3], "language": "Bengali"} for i in range(30)]
    split, info = language_held_out_split(recs, "Bengali")
    assert info["held_out_language"] == "Bengali"
    assert all(r["language"] == "Bengali" for r in split["test"])
    assert all(r["language"] == "Hindi" for r in split["train"])


def test_metrics_perfect_and_chance():
    y = ["FAVOUR", "AGAINST", "NONE", "FAVOUR"]
    perfect = compute_metrics(y, y)
    assert perfect["macro_f1"] == 1.0 and perfect["accuracy"] == 1.0
    chance = compute_metrics(y, ["NONE"] * 4)
    assert chance["macro_f1"] < 0.5


def test_confusion_matrix_shape():
    y = ["FAVOUR", "AGAINST", "NONE"]
    m = compute_metrics(y, y)
    assert len(m["confusion_matrix"]) == 3 and len(m["confusion_matrix"][0]) == 3


def test_bootstrap_ci_brackets_point():
    y = ["FAVOUR", "AGAINST", "NONE"] * 20
    p = ["FAVOUR", "AGAINST", "NONE"] * 19 + ["NONE", "FAVOUR", "AGAINST"]
    ci = bootstrap_macro_f1(y, p, n_boot=200)
    assert ci["ci95_low"] <= ci["macro_f1"] <= ci["ci95_high"]


def test_mcnemar_identical():
    y = ["FAVOUR", "AGAINST"]
    assert mcnemar_test(y, y, y)["p_value"] == 1.0


def test_per_language_report():
    recs = [{"language": "Hindi"}, {"language": "Bengali"}]
    rep = per_language_report(recs, ["FAVOUR", "NONE"], ["FAVOUR", "NONE"])
    assert set(rep) == {"Hindi", "Bengali"}
    assert rep["Hindi"]["accuracy"] == 1.0
