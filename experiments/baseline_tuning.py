"""Tune local TF-IDF baselines using only the supplied training partition.

Run with /opt/anaconda3/envs/dml/bin/python experiments/baseline_tuning.py.
The script deliberately never opens a test partition. See the generated README
for the distinction between model-selection CV and independent evaluation.
"""

from __future__ import annotations

import os

# Set before scientific imports so each of the two CV workers uses one thread.
for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
)
from sklearn.pipeline import Pipeline


SEED = 42
LABELS = [0, 1, 2, 3, 4]
REPRESENTATIONS = ["StemmedText", "CleanText"]


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=json_value) + "\n")


def pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "quadratic_weighted_kappa": cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic"),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
    }


def run(train_path: Path, output_dir: Path) -> None:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(train_path)
    required = {"PhraseId", "SentenceId", "Phrase", "Sentiment", *REPRESENTATIONS}
    if not required.issubset(frame.columns):
        raise ValueError(f"Training data missing columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isnull().any().any():
        raise ValueError("Training input contains missing required values")
    if sorted(frame.Sentiment.unique().tolist()) != LABELS:
        raise ValueError("Expected the five sentiment labels 0 through 4")
    if frame.SentenceId.duplicated().any():
        raise ValueError("Repeated SentenceId requires grouped CV instead of stratified row CV")
    for column in REPRESENTATIONS:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Empty text in {column}")
        if frame[column].duplicated().any():
            raise ValueError(f"Duplicate training texts in {column} require duplicate-aware CV")

    y = frame.Sentiment.astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # Reuse identical, saved folds for every representation and candidate.
    splits = list(cv.split(frame, y))
    fold_assignment = np.full(len(frame), -1, dtype=int)
    for fold, (_, validation_idx) in enumerate(splits):
        fold_assignment[validation_idx] = fold
    pd.DataFrame({
        "row_index": np.arange(len(frame)),
        "PhraseId": frame.PhraseId,
        "SentenceId": frame.SentenceId,
        "Sentiment": y,
        "cv_fold": fold_assignment,
    }).to_csv(output_dir / "cv_folds.csv", index=False)

    scoring = {"macro_f1": "f1_macro", "accuracy": "accuracy"}
    baseline = pipeline().set_params(lr__max_iter=1000)
    with parallel_backend("loky", inner_max_num_threads=1):
        original_scores = cross_validate(
            baseline, frame.StemmedText, y, cv=splits,
            scoring=scoring, n_jobs=2, error_score="raise",
        )
    baseline_result = {
        "description": "Original default StemmedText TF-IDF + LogisticRegression baseline",
        "parameters": {
            "representation": "StemmedText", "tfidf": "sklearn defaults",
            "lr_max_iter": 1000, "lr_C": 1.0, "lr_class_weight": None,
            "random_state": SEED,
        },
        "folds": {
            "macro_f1": original_scores["test_macro_f1"].tolist(),
            "accuracy": original_scores["test_accuracy"].tolist(),
        },
        "mean_macro_f1": float(original_scores["test_macro_f1"].mean()),
        "std_macro_f1": float(original_scores["test_macro_f1"].std()),
        "mean_accuracy": float(original_scores["test_accuracy"].mean()),
        "std_accuracy": float(original_scores["test_accuracy"].std()),
    }
    write_json(output_dir / "original_baseline_cv.json", baseline_result)
    print(f"Original baseline CV macro-F1: {baseline_result['mean_macro_f1']:.4f}", flush=True)

    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [True],
        "lr__C": [0.1, 1, 10],
        "lr__class_weight": [None, "balanced"],
    }
    tables = []
    selected_search = None
    selected_representation = None
    for representation in REPRESENTATIONS:
        search = GridSearchCV(
            pipeline(), grid, cv=splits, scoring=scoring, refit="macro_f1",
            n_jobs=2, return_train_score=True, error_score="raise",
        )
        with parallel_backend("loky", inner_max_num_threads=1):
            search.fit(frame[representation], y)
        candidate_table = pd.DataFrame(search.cv_results_)
        candidate_table.insert(0, "representation", representation)
        tables.append(candidate_table)
        print(f"{representation}: best CV macro-F1 {search.best_score_:.4f}; {search.best_params_}", flush=True)
        if selected_search is None or search.best_score_ > selected_search.best_score_:
            selected_search = search
            selected_representation = representation

    candidates = pd.concat(tables, ignore_index=True)
    candidates["global_rank_macro_f1"] = candidates.mean_test_macro_f1.rank(ascending=False, method="min").astype(int)
    candidates.to_csv(output_dir / "cv_candidates.csv", index=False)
    assert selected_search is not None and selected_representation is not None
    chosen_model = selected_search.best_estimator_
    # These are OOF predictions for the selected hyperparameters, not nested CV.
    # Selection has already seen every fold's score; report that limitation.
    with parallel_backend("loky", inner_max_num_threads=1):
        oof = cross_val_predict(
            chosen_model, frame[selected_representation], y,
            cv=splits, n_jobs=2, method="predict",
        )
    oof_frame = frame[["PhraseId", "SentenceId", "Phrase", "Sentiment"]].copy()
    oof_frame.insert(0, "row_index", np.arange(len(frame)))
    oof_frame["cv_fold"] = fold_assignment
    oof_frame["predicted_sentiment"] = oof
    oof_frame.to_csv(output_dir / "selected_model_oof_predictions.csv", index=False)
    selected_summary = {
        "representation": selected_representation,
        "best_parameters": selected_search.best_params_,
        "selection_metric": "mean five-fold validation macro-F1",
        "mean_cv_macro_f1": selected_search.best_score_,
        "std_cv_macro_f1": selected_search.cv_results_["std_test_macro_f1"][selected_search.best_index_],
        "mean_cv_accuracy": selected_search.cv_results_["mean_test_accuracy"][selected_search.best_index_],
        "candidate_count": len(candidates),
        "fold_count": len(splits),
        "random_seed": SEED,
        "selection_bias_warning": (
            "The same CV folds selected representation and hyperparameters. "
            "Their CV and selected-model OOF scores are optimistic model-selection "
            "diagnostics, not independent estimates. The supplied test partition "
            "has not been opened by this script."
        ),
        "oof_metrics": metrics(y, oof),
        "oof_classification_report": classification_report(
            y, oof, labels=LABELS, zero_division=0, output_dict=True,
        ),
        "oof_confusion_matrix": confusion_matrix(y, oof, labels=LABELS).tolist(),
        "training_rows": len(frame),
        "train_class_counts": frame.Sentiment.value_counts().sort_index().to_dict(),
        "model_file": "selected_model.joblib",
    }
    write_json(output_dir / "selected_configuration.json", selected_summary)
    # GridSearchCV already refits best_estimator_ on every supplied training row.
    joblib.dump({
        "pipeline": chosen_model,
        "representation": selected_representation,
        "label_order": LABELS,
        "training_rows": len(frame),
        "source_train_file": str(train_path.resolve()),
        "random_seed": SEED,
        "selected_parameters": selected_search.best_params_,
    }, output_dir / "selected_model.joblib")
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]
    }
    write_json(output_dir / "run_metadata.json", {
        "python": platform.python_version(), "dependencies": versions,
        "elapsed_seconds": time.monotonic() - started,
        "training_file": str(train_path.resolve()),
        "n_jobs": 2, "threads_per_worker": 1,
    })
    (output_dir / "README.md").write_text(
        "# Baseline tuning\n\n"
        "Run from the project root:\n\n"
        "```sh\n/opt/anaconda3/envs/dml/bin/python experiments/baseline_tuning.py\n```\n\n"
        "This experiment reads only `Cleaned data/cleaned_train.csv`. It never opens "
        "the supplied test partition. Five stratified folds (shuffle seed 42) are "
        "shared by all candidates. TF-IDF is fitted inside each training fold. "
        "Every training row has a unique SentenceId and unique text, checked before CV.\n\n"
        "The 48 candidates compare StemmedText and CleanText using "
        "C in {0.1, 1, 10}, unigram/bigram features, min_df in {1, 2}, "
        "class_weight in {None, balanced}, and sublinear_tf=True. "
        "The original default StemmedText baseline is evaluated separately.\n\n"
        "`cv_candidates.csv` includes all candidate and fold scores. "
        "`cv_folds.csv` records row assignments. `selected_configuration.json` "
        "records selection, metrics, and the confusion matrix. "
        "`selected_model_oof_predictions.csv` contains selected-model out-of-fold "
        "predictions. Because the same CV folds selected the model, these scores "
        "have selection optimism; they are not an independent final evaluation.\n\n"
        "`selected_model.joblib` contains a dictionary with `pipeline` and "
        "`representation`; the pipeline is fitted on all 511 training rows. "
        "For later evaluation, load the dictionary and call "
        "`bundle['pipeline'].predict(frame[bundle['representation']])`. "
        "Use only trusted joblib files.\n\n"
        "Dependencies: numpy, pandas, scipy, scikit-learn, joblib, threadpoolctl. "
        "Exact installed versions and runtime are in `run_metadata.json`. "
        "Parallelism is limited to two workers and one numerical thread each.\n"
    )
    print(f"Selected {selected_representation}; OOF macro-F1 {selected_summary['oof_metrics']['macro_f1']:.4f}", flush=True)
    print(f"Saved outputs to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=root / "Cleaned data" / "cleaned_train.csv")
    parser.add_argument("--output", type=Path, default=root / "experiments" / "results" / "baseline")
    arguments = parser.parse_args()
    run(arguments.train, arguments.output)
