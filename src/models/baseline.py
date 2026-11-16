"""Traditional ML baselines: TF-IDF + linear models.

Reference points only - the scientific question is whether contextual
multilingual encoders (teacher) and a distilled student beat these. Everything is
macro-F1 driven, class-weighted, and evaluated on a *group-stratified* split so
the numbers are not inflated by the 253 duplicate sentences found in the audit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

FeatureVariant = Literal["word", "char", "combined"]


@dataclass
class TfidfConfig:
    feature: FeatureVariant = "combined"
    model: Literal["logreg", "svm"] = "logreg"
    C: float = 1.0
    class_weight: str | None = "balanced"
    word_ngram: tuple[int, int] = (1, 2)
    char_ngram: tuple[int, int] = (2, 5)
    max_word_features: int = 50_000
    max_char_features: int = 100_000
    lowercase: bool = True
    min_df: int = 1
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "feature": self.feature, "model": self.model, "C": self.C,
            "class_weight": self.class_weight, "word_ngram": list(self.word_ngram),
            "char_ngram": list(self.char_ngram), "lowercase": self.lowercase, "min_df": self.min_df,
        }


class TfidfStanceClassifier:
    """Word / char / combined TF-IDF feeding LogisticRegression or LinearSVC."""

    def __init__(self, config: TfidfConfig | None = None):
        self.config = config or TfidfConfig()
        self.vec_word: TfidfVectorizer | None = None
        self.vec_char: TfidfVectorizer | None = None
        self.clf = None
        self.classes_: list[str] = []

    # ---------------- features ----------------
    def _fit_vectorizers(self, texts: list[str]) -> None:
        c = self.config
        if c.feature in ("word", "combined"):
            self.vec_word = TfidfVectorizer(
                analyzer="word", ngram_range=c.word_ngram, max_features=c.max_word_features,
                lowercase=c.lowercase, min_df=c.min_df, sublinear_tf=True,
            )
            self.vec_word.fit(texts)
        if c.feature in ("char", "combined"):
            self.vec_char = TfidfVectorizer(
                analyzer="char_wb", ngram_range=c.char_ngram, max_features=c.max_char_features,
                lowercase=c.lowercase, min_df=c.min_df, sublinear_tf=True,
            )
            self.vec_char.fit(texts)

    def _transform(self, texts: list[str]):
        parts = []
        if self.vec_word is not None:
            parts.append(self.vec_word.transform(texts))
        if self.vec_char is not None:
            parts.append(self.vec_char.transform(texts))
        if len(parts) == 1:
            return parts[0]
        return hstack(parts).tocsr()

    # ---------------- model ----------------
    def _build_clf(self):
        c = self.config
        if c.model == "logreg":
            # lbfgs (not liblinear): liblinear is binary-only and cannot fit the
            # three-class problem under current scikit-learn.
            return LogisticRegression(
                C=c.C, class_weight=c.class_weight, max_iter=3000, solver="lbfgs", random_state=c.seed,
            )
        base = LinearSVC(C=c.C, class_weight=c.class_weight, random_state=c.seed, max_iter=5000)
        # LinearSVC has no predict_proba; calibrate so KD/probability outputs exist.
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfStanceClassifier":
        self._fit_vectorizers(texts)
        X = self._transform(texts)
        self.clf = self._build_clf()
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.clf.predict(self._transform(texts)))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._transform(texts)
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)
        # fallback: margin -> softmax over classes
        d = self.clf.decision_function(X)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def n_features(self) -> int:
        n = 0
        if self.vec_word is not None:
            n += len(self.vec_word.vocabulary_)
        if self.vec_char is not None:
            n += len(self.vec_char.vocabulary_)
        return n

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config.to_dict(), "model": self}, p)
        return p

    @staticmethod
    def load(path: str | Path) -> "TfidfStanceClassifier":
        return joblib.load(Path(path))["model"]

    def top_features(self, label: str, k: int = 15) -> list[str]:
        """Most positive-weighted features for a class (explorability of the baseline)."""
        if not hasattr(self.clf, "coef_"):
            return []
        names = []
        if self.vec_word is not None:
            names += list(self.vec_word.get_feature_names_out())
        if self.vec_char is not None:
            names += list(self.vec_char.get_feature_names_out())
        try:
            idx = self.classes_.index(label)
        except ValueError:
            return []
        coefs = self.clf.coef_
        row = coefs[idx] if coefs.shape[0] > 1 else coefs[0]
        order = np.argsort(row)[::-1][:k]
        return [names[i] for i in order]


def majority_predictor(y_train: list[str], n: int) -> list[str]:
    from collections import Counter

    return [Counter(y_train).most_common(1)[0][0]] * n
