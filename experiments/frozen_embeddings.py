#!/usr/bin/env python3
"""Offline pretrained-encoder comparison with nested, train-only LR tuning.

Run with /opt/anaconda3/envs/dml/bin/python experiments/frozen_embeddings.py.
Only --extract-test reads test text; that mode never reads test labels or tunes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "experiments/features"
RESULTS = ROOT / "experiments/results/frozen"
SEED = 42
MODELS = {
    "minilm": {"id": "sentence-transformers/all-MiniLM-L6-v2", "pooling": "mean", "max_length": 128,
                "description": "Sentence-transformer pretrained with sentence-pair objectives"},
    "bge_small": {"id": "BAAI/bge-small-en-v1.5", "pooling": "cls", "max_length": 128,
                   "description": "Sentence/retrieval pretrained BGE; no retrieval query instruction for review classification"},
    "bert_base": {"id": "google-bert/bert-base-uncased", "pooling": "mean", "max_length": 128,
                   "description": "Generic masked-language encoder; mean pooling is a baseline, not a sentence-trained model"},
    "distilbert": {"id": "distilbert-base-uncased", "pooling": "mean", "max_length": 128,
                    "description": "Generic distilled language encoder; mean pooling is a baseline, not a sentence-trained model"},
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_for(model_id: str) -> Path:
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    directory = cache / ("models--" + model_id.replace("/", "--"))
    reference = directory / "refs/main"
    if not reference.exists():
        raise FileNotFoundError(f"Offline model main reference not cached: {reference}")
    snapshot = directory / "snapshots" / reference.read_text().strip()
    if not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Offline model weights missing: {snapshot}")
    return snapshot


@torch.no_grad()
def encode(texts: list[str], tokenizer, model, spec, batch_size: int, device: str) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=spec["max_length"], return_tensors="pt")
        tokenized = {name: value.to(device) for name, value in tokenized.items()}
        output = model(**tokenized).last_hidden_state
        if spec["pooling"] == "cls":
            pooled = output[:, 0]
        else:
            mask = tokenized["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        rows.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def extract(key: str, frame: pd.DataFrame, split: str, batch_size: int, device: str) -> np.ndarray:
    spec = MODELS[key]
    snapshot = snapshot_for(spec["id"])
    output_path = FEATURES / f"{key}_{split}.npy"
    meta_path = FEATURES / f"{key}_{split}_metadata.json"
    text_digest = hashlib.sha256("\n".join(frame.Phrase).encode("utf8")).hexdigest()
    if output_path.exists() and meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if (previous.get("text_sha256") == text_digest and previous.get("snapshot") == snapshot.name
                and previous.get("pooling") == spec["pooling"] and previous.get("max_length") == spec["max_length"]):
            print(f"Reusing verified {key} {split} embeddings", flush=True)
            array = np.load(output_path)
            if sha256(output_path) != previous["embedding_file_sha256"]:
                raise ValueError(f"Embedding integrity mismatch: {output_path}")
            return array
    start = time.time()
    print(f"Extracting {key} {split} on {device} from cached snapshot {snapshot.name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True).to(device).eval()
    texts = frame.Phrase.astype(str).tolist()
    lengths = [len(x) for x in tokenizer(texts, truncation=False, padding=False)["input_ids"]]
    array = encode(texts, tokenizer, model, spec, batch_size, device)
    repeat = encode(texts[:3], tokenizer, model, spec, batch_size, device)
    delta = float(np.max(np.abs(array[:3] - repeat)))
    norms = np.linalg.norm(array, axis=1)
    assert np.isfinite(array).all() and np.max(np.abs(norms - 1)) < 1e-5
    assert delta < 1e-5, f"Encoder repeatability failure: {delta}"
    FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    cols = [c for c in ["PhraseId", "SentenceId", "Sentiment"] if c in frame.columns]
    row_frame = frame[cols].copy()
    row_frame.insert(0, "row_index", np.arange(len(frame)))
    row_frame.to_csv(FEATURES / f"{key}_{split}_rows.csv", index=False)
    provenance = dict(spec, snapshot=snapshot.name, local_snapshot=str(snapshot),
                      split=split, samples=len(frame), dimensions=array.shape[1], dtype=str(array.dtype),
                      text_sha256=text_digest, source_file_sha256=sha256(ROOT / f"{split}.csv"),
                      embedding_file_sha256=sha256(output_path), device=device, torch_threads=torch.get_num_threads(),
                      torch_version=torch.__version__, transformers_version=transformers.__version__,
                      numpy_version=np.__version__, sklearn_version=sklearn.__version__, python_version=platform.python_version(),
                      row_order="Original CSV order; see adjacent rows CSV", normalization="L2", raw_text=True,
                      includes_special_tokens_in_mean=(spec["pooling"] == "mean"),
                      maximum_observed_tokens=max(lengths), truncated_rows=sum(x > spec["max_length"] for x in lengths),
                      smoke_checks={"all_finite": True, "max_unit_norm_error": float(np.max(np.abs(norms - 1))),
                                    "repeatability_max_abs_delta_first3": delta},
                      files_sha256={name: sha256(snapshot / name) for name in
                                    ["model.safetensors", "config.json", "tokenizer.json"] if (snapshot / name).is_file()},
                      extraction_seconds=time.time() - start)
    write_json(meta_path, provenance)
    del model, tokenizer
    print(f"Saved {key}: {array.shape}; {time.time() - start:.1f}s", flush=True)
    return array


def metrics(y, predicted, probabilities=None):
    value = {"accuracy": accuracy_score(y, predicted), "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
             "class_0_recall": recall_score(y, predicted, labels=[0], average=None, zero_division=0)[0],
             "ordinal_mae": float(np.mean(np.abs(y - predicted))),
             "within_one_accuracy": float(np.mean(np.abs(y - predicted) <= 1)),
             "confusion_matrix": confusion_matrix(y, predicted, labels=np.arange(5)).tolist(),
             "classification_report": classification_report(y, predicted, labels=np.arange(5), output_dict=True, zero_division=0)}
    if probabilities is not None:
        value["log_loss"] = log_loss(y, probabilities, labels=np.arange(5))
    return value


def search_classifier(X, y, random_state=SEED):
    grid = GridSearchCV(LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED),
                        {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]},
                        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"}, refit="macro_f1",
                        cv=StratifiedKFold(3, shuffle=True, random_state=random_state),
                        n_jobs=1, return_train_score=True, error_score="raise")
    grid.fit(X, y)
    return grid


def evaluate_train(key: str, X: np.ndarray, frame: pd.DataFrame):
    output = RESULTS / key
    output.mkdir(parents=True, exist_ok=True)
    y = frame.Sentiment.to_numpy(dtype=int)
    probs = np.zeros((len(y), 5), dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=int)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
        grid = search_classifier(X[train_idx], y[train_idx], SEED + fold)
        prob = grid.predict_proba(X[val_idx])
        probs[val_idx] = prob
        fold_ids[val_idx] = fold
        pd.DataFrame(grid.cv_results_).to_csv(output / f"fold_{fold}_inner_search.csv", index=False)
        fold_result = {"fold": fold, "train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist(),
                       "best_params": grid.best_params_, "inner_best_macro_f1": grid.best_score_,
                       "metrics": metrics(y[val_idx], grid.predict(X[val_idx]), prob)}
        folds.append(fold_result)
        print(f"{key} outer fold {fold}: macroF1={fold_result['metrics']['macro_f1']:.4f}, {grid.best_params_}", flush=True)
    predictions = probs.argmax(1)
    rows = frame[["PhraseId", "SentenceId", "Sentiment"]].copy()
    rows.insert(0, "row_index", np.arange(len(y)))
    rows["outer_fold"] = fold_ids
    rows["prediction"] = predictions
    for label in range(5):
        rows[f"prob_{label}"] = probs[:, label]
    rows.to_csv(output / "nested_oof_predictions.csv", index=False)
    np.save(output / "nested_oof_probabilities.npy", probs)
    final = search_classifier(X, y)
    pd.DataFrame(final.cv_results_).to_csv(output / "full_train_inner_search.csv", index=False)
    joblib.dump(final.best_estimator_, output / "classifier.joblib")
    result = {"model": key, "encoder": MODELS[key], "seed": SEED,
              "protocol": "5 stratified shuffled outer folds; 3 stratified shuffled inner folds; macro-F1 selection; no test use",
              "tuning_grid": {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]},
              "feature_transform": "Unit-L2 encoder vectors; no scaling or supervised transform",
              "nested_oof": metrics(y, predictions, probs), "folds": folds,
              "final_params": final.best_params_, "full_train_cv_selection_macro_f1": final.best_score_,
              "final_train_metrics": metrics(y, final.predict(X), final.predict_proba(X)),
              "classifier_sha256": sha256(output / "classifier.joblib"),
              "warnings": ["Do not use training fit or full-train tuning CV as an unbiased performance estimate.",
                           "Model-family selection from these nested estimates introduces additional selection optimism; reserve test for final comparison."]}
    write_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--extract-test", action="store_true", help="Final evaluation only: extract test text with usecols, predict; never access labels")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.extract_test:
        frame = pd.read_csv(ROOT / "test.csv", usecols=["PhraseId", "SentenceId", "Phrase"])
        for key in args.models:
            X = extract(key, frame, "test", args.batch_size, args.device)
            model = joblib.load(RESULTS / key / "classifier.joblib")
            rows = frame[["PhraseId", "SentenceId"]].copy()
            rows.insert(0, "row_index", np.arange(len(frame)))
            rows["prediction"] = model.predict(X)
            for label, column in enumerate(model.predict_proba(X).T):
                rows[f"prob_{label}"] = column
            rows.to_csv(RESULTS / key / "test_predictions.csv", index=False)
        return
    frame = pd.read_csv(ROOT / "train.csv")
    assert frame.Phrase.notna().all() and set(frame.Sentiment) == set(range(5))
    results = []
    for key in args.models:
        X = extract(key, frame, "train", args.batch_size, args.device)
        results.append(evaluate_train(key, X, frame))
    summary = []
    for path in RESULTS.glob("*/metrics.json"):
        result = json.loads(path.read_text())
        summary.append({"model": result["model"], **{k: result["nested_oof"][k] for k in
                        ["accuracy", "macro_f1", "class_0_recall", "ordinal_mae", "within_one_accuracy", "log_loss"]},
                        "final_C": result["final_params"]["C"], "final_class_weight": result["final_params"]["class_weight"]})
    pd.DataFrame(summary).sort_values("macro_f1", ascending=False).to_csv(RESULTS / "comparison.csv", index=False)
    print(pd.DataFrame(summary).sort_values("macro_f1", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
