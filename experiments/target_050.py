"""Training-only, nested-CV search for >=0.50 macro-F1 (never reads test.csv).

Based on BigdataGP cached frozen sentence vectors and raw review text.
Each candidate has inner 3-fold hyperparameter selection within outer 5-fold.
Outer results are diagnostic; selecting among many outer results is optimistic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "experiments" / "features"


def read_data(train_path: Path):
    df = pd.read_csv(train_path)
    if not {"PhraseId", "Phrase", "Sentiment"}.issubset(df):
        raise ValueError("Need PhraseId, Phrase, Sentiment")
    if df[["PhraseId", "Phrase", "Sentiment"]].isna().any().any():
        raise ValueError("Missing required data")
    if not df.PhraseId.is_unique:
        raise ValueError("Nonunique PhraseId")
    y = df.Sentiment.to_numpy(dtype=int)
    if set(y.tolist()) != set(range(5)):
        raise ValueError("Expected five classes")
    return df, y


def frozen_features(df, name):
    rows = pd.read_csv(FEATURE_DIR / f"{name}_train_rows.csv")
    if not np.array_equal(df.PhraseId.to_numpy(), rows.PhraseId.to_numpy()):
        raise ValueError(f"ID ordering mismatch: {name}")
    if "Sentiment" in rows and not np.array_equal(df.Sentiment.to_numpy(), rows.Sentiment.to_numpy()):
        raise ValueError(f"Label mismatch: {name}")
    X = np.load(FEATURE_DIR / f"{name}_train.npy", allow_pickle=False)
    if len(X) != len(df) or not np.isfinite(X).all():
        raise ValueError(f"Embedding shape/finite mismatch: {name}")
    return X.astype("float64")


class OrdinalThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Four cumulative binary logistic models; monotonicity via sorting."""

    def __init__(self, C=1., class_weight=None):
        self.C = C
        self.class_weight = class_weight

    def fit(self, X, y):
        self.classes_ = np.arange(5)
        self.models_ = []
        for threshold in range(4):
            binary = np.asarray(y) > threshold
            clf = LogisticRegression(
                C=self.C, max_iter=1500, class_weight=self.class_weight, solver="lbfgs"
            )
            clf.fit(X, binary)
            self.models_.append(clf)
        return self

    def predict_proba(self, X):
        survival = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        survival = np.minimum.accumulate(survival, axis=1)
        probs = np.column_stack([
            1-survival[:, 0],
            survival[:, 0]-survival[:, 1],
            survival[:, 1]-survival[:, 2],
            survival[:, 2]-survival[:, 3],
            survival[:, 3]
        ])
        return np.clip(probs, 0, 1) / np.clip(probs.sum(1, keepdims=True), 1e-12, None)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def inputs_for(df):
    bge = frozen_features(df, "bge_small")
    mini = frozen_features(df, "minilm")
    bert = frozen_features(df, "bert_base")
    distil = frozen_features(df, "distilbert")
    raw = df.Phrase.astype(str).tolist()
    def norm(x):
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x
    return {
        "bge": norm(bge),
        "bge_minilm": norm(np.hstack([bge, mini])),
        "bge_bert": norm(np.hstack([bge, bert])),
        "bge_all": norm(np.hstack([bge, mini, bert, distil])),
        "raw": np.array(raw, dtype=object),
    }


def make_pipeline(kind):
    if kind == "bge_rbf":
        return ("bge", SVC(kernel="rbf", decision_function_shape="ovr"), {
            "C": [0.3, 1, 3, 10, 30], "gamma": ["scale", 0.005, 0.02],
            "class_weight": [None, "balanced"]})
    if kind == "bge_linear":
        return ("bge", LogisticRegression(max_iter=2000), {
            "C": [0.1, 1, 10, 100], "class_weight": [None, "balanced"]})
    if kind == "bge_ordinal":
        return ("bge", OrdinalThresholdClassifier(), {
            "C": [0.03, 0.3, 3, 30], "class_weight": [None, "balanced"]})
    if kind == "bge_minilm_rbf":
        return ("bge_minilm", SVC(kernel="rbf"), {
            "C": [0.3, 1, 3, 10], "gamma": ["scale", 0.01],
            "class_weight": [None, "balanced"]})
    if kind == "bge_bert_rbf":
        return ("bge_bert", SVC(kernel="rbf"), {
            "C": [0.3, 1, 3, 10], "gamma": ["scale", 0.01],
            "class_weight": [None, "balanced"]})
    if kind == "bge_all_linear":
        return ("bge_all", LogisticRegression(max_iter=2000), {
            "C": [0.1, 1, 10, 100], "class_weight": [None, "balanced"]})
    if kind == "bge_all_rbf":
        return ("bge_all", SVC(kernel="rbf"), {
            "C": [0.3, 1, 3, 10], "gamma": ["scale", 0.01],
            "class_weight": [None, "balanced"]})
    if kind == "wordchar_svm":
        vectorizer = ColumnTransformer([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                     strip_accents="unicode", sublinear_tf=True),
             "Phrase"),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                     min_df=2, sublinear_tf=True),
             "Phrase"),
        ], sparse_threshold=1.0)
        return ("frame", Pipeline([
            ("features", vectorizer),
            ("classifier", LinearSVC(max_iter=5000))
        ]), {"classifier__C": [0.1, 1, 10],
              "classifier__class_weight": [None, "balanced"]})
    if kind == "wordchar_bge":
        # Input is a DataFrame; dense BGE columns stored as scalar float cols.
        vectorizer = ColumnTransformer([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                     sublinear_tf=True), "Phrase"),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                     min_df=2, sublinear_tf=True), "Phrase"),
            ("bge", Pipeline([("scale", FunctionTransformer(
                lambda x: x * 4.0, validate=False
            ))]), [f"bge_{i}" for i in range(384)])
        ], sparse_threshold=1.0)
        return ("frame", Pipeline([
            ("features", vectorizer), ("classifier", LinearSVC(max_iter=6000))
        ]), {"classifier__C": [0.03, 0.3, 3, 30],
              "classifier__class_weight": [None, "balanced"]})
    raise ValueError(kind)


def nested_evaluate(df, y, data, kind, out, seed=42):
    selector, estimator, grid = make_pipeline(kind)
    if selector == "frame":
        X = df.copy()
        if kind == "wordchar_bge":
            for i in range(384):
                X[f"bge_{i}"] = data["bge"][:, i]
            # Avoid fragmented DataFrame performance warning in downstream code.
            X = X.copy()
    else:
        X = data[selector]
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.full(len(y), -1, dtype=int)
    fold_data = []
    for fold, (train_ids, valid_ids) in enumerate(outer.split(X, y)):
        x_train = X.iloc[train_ids] if hasattr(X, "iloc") else X[train_ids]
        x_valid = X.iloc[valid_ids] if hasattr(X, "iloc") else X[valid_ids]
        inner = StratifiedKFold(3, shuffle=True, random_state=seed + fold + 1)
        search = GridSearchCV(estimator, grid, scoring="f1_macro", refit=True,
                              cv=inner, n_jobs=1, error_score="raise")
        search.fit(x_train, y[train_ids])
        pred = search.predict(x_valid).astype(int)
        oof[valid_ids] = pred
        f1 = f1_score(y[valid_ids], pred, average="macro", zero_division=0)
        fold_data.append({"fold": fold, "inner_cv_macro_f1": search.best_score_,
                          "outer_macro_f1": f1,
                          "outer_accuracy": accuracy_score(y[valid_ids], pred),
                          "best_params": search.best_params_})
        print(f"{kind} fold={fold} OOF_macroF1={f1:.4f} inner={search.best_score_:.4f}", flush=True)
    if np.any(oof < 0):
        raise AssertionError("Not all rows received out-of-fold predictions")
    score = {"name": kind, "accuracy": accuracy_score(y, oof),
             "macro_f1": f1_score(y, oof, average="macro", zero_division=0),
             "recall": recall_score(y, oof, labels=np.arange(5), average=None,
                                     zero_division=0).tolist(),
             "confusion": confusion_matrix(y, oof, labels=np.arange(5)).tolist(),
             "folds": fold_data}
    pd.DataFrame({"PhraseId": df.PhraseId, "true": y, "prediction": oof,
                  "model": kind}).to_csv(out / f"{kind}_oof.csv", index=False)
    (out / f"{kind}.json").write_text(json.dumps(score, indent=2))
    print(f"FINAL {kind} accuracy={score['accuracy']:.4f} macroF1={score['macro_f1']:.4f}", flush=True)
    return score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=str(ROOT / "train.csv"))
    p.add_argument("--models", nargs="+", default=[
        "bge_rbf", "bge_linear", "bge_ordinal", "bge_minilm_rbf",
        "bge_bert_rbf", "bge_all_linear", "bge_all_rbf",
        "wordchar_svm", "wordchar_bge"])
    p.add_argument("--out", default=str(ROOT / "experiments" / "results" / "target_050"))
    args = p.parse_args()
    df, y = read_data(Path(args.train))
    features = inputs_for(df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    for model in args.models:
        scores.append(nested_evaluate(df, y, features, model, out))
        pd.DataFrame([{k:v for k,v in s.items() if k not in ("recall","confusion","folds")}
                      for s in scores]).sort_values("macro_f1", ascending=False).to_csv(
                          out / "leaderboard.csv", index=False)
    (out / "PROTOCOL.txt").write_text(
        "Five-fold nested selection on 511 train.csv reviews only; 3-fold inner model tuning.\n"
        "No test.csv read; selecting among model-family outer CV scores introduces optimism.\n"
        "Report class recalls and complete OOF predictions; 0.50 on outer scores is not\n"
        "proof of 0.50 on held-out test.\n")
    print(pd.read_csv(out / "leaderboard.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
