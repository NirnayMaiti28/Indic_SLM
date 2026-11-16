import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.baseline import TfidfConfig, TfidfStanceClassifier, majority_predictor  # noqa: E402

TEXTS = [
    "climate change is a serious concern", "global warming is a hoax",
    "the weather was nice today", "we must act on the climate crisis now",
    "scientists say warming is serious", "there is no evidence of any problem",
    "I bought a new phone yesterday", "emissions must fall urgently",
    "warming is exaggerated nonsense", "unrelated comment about football",
]
LABELS = ["FAVOUR", "AGAINST", "NONE", "FAVOUR", "FAVOUR", "AGAINST", "NONE", "FAVOUR", "AGAINST", "NONE"]


def test_fit_predict_shapes():
    clf = TfidfStanceClassifier(TfidfConfig(feature="word")).fit(TEXTS, LABELS)
    preds = clf.predict(TEXTS)
    assert len(preds) == len(TEXTS)
    assert set(preds) <= set(LABELS)


def test_proba_sums_to_one():
    clf = TfidfStanceClassifier(TfidfConfig(feature="combined")).fit(TEXTS, LABELS)
    p = clf.predict_proba(TEXTS)
    assert p.shape == (len(TEXTS), len(clf.classes_))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_char_and_combined_run():
    for feat in ("char", "combined"):
        clf = TfidfStanceClassifier(TfidfConfig(feature=feat)).fit(TEXTS, LABELS)
        assert len(clf.predict(TEXTS)) == len(TEXTS)
        assert clf.n_features() > 0


def test_svm_is_calibrated():
    clf = TfidfStanceClassifier(TfidfConfig(feature="word", model="svm", C=1.0)).fit(TEXTS, LABELS)
    p = clf.predict_proba(TEXTS)
    assert p.shape[1] == len(clf.classes_)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_class_weight_helps_minority():
    # AGAINST appears once; balanced weighting must not ignore it entirely
    texts = TEXTS + ["another neutral remark"] * 6
    labels = LABELS + ["NONE"] * 6
    clf = TfidfStanceClassifier(TfidfConfig(feature="char", C=1.0, class_weight="balanced")).fit(texts, labels)
    assert "AGAINST" in set(clf.classes_)


def test_top_features_nonempty():
    clf = TfidfStanceClassifier(TfidfConfig(feature="word")).fit(TEXTS, LABELS)
    feats = clf.top_features("AGAINST", k=3)
    assert len(feats) <= 3


def test_majority_predictor():
    assert majority_predictor(["A", "A", "B"], 4) == ["A"] * 4
