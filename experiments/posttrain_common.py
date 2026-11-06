"""Shared train-only evaluation and LLM label handling for post-training trials.

Never import or read test.csv. Sentiment labels are exactly 0,1,2,3,4.
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
LABELS = np.arange(5, dtype=int)
SENTIMENT_NAMES = (
    "very negative", "somewhat negative", "neutral or mixed",
    "somewhat positive", "very positive",
)


def load_training_csv(path=None):
    path = Path(path) if path else ROOT / "train.csv"
    frame = pd.read_csv(path)
    fields = ("PhraseId", "Phrase", "Sentiment")
    if not set(fields).issubset(frame.columns):
        raise ValueError("Expected PhraseId, Phrase, Sentiment")
    if frame[list(fields)].isna().any().any() or not frame.PhraseId.is_unique:
        raise ValueError("Null entries or repeated PhraseId in training data")
    y = frame.Sentiment.to_numpy(dtype=np.int64)
    if set(y.tolist()) != set(LABELS.tolist()):
        raise ValueError("Must have all five labels 0,1,2,3,4")
    return frame, y


def splits(y, folds=5, seed=42):
    if folds < 2 or np.bincount(y, minlength=5).min() < folds:
        raise ValueError("Invalid fold count or too few class members")
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(
        np.arange(len(y)), y
    ))


def training_fewshots(frame, y, train_indices, *, seed, per_class=1):
    """Only uses training-fold IDs; selects equal class counts."""
    if per_class < 1:
        raise ValueError("per_class must be >= 1")
    rng = np.random.default_rng(seed)
    result = []
    for label in LABELS:
        available = np.asarray(train_indices)[y[train_indices] == label]
        if len(available) < per_class:
            raise ValueError("Too few training-fold examples for this class")
        selected = rng.choice(available, size=per_class, replace=False)
        for index in selected:
            result.append((str(frame.Phrase.iloc[int(index)]), int(label), int(index)))
    rng.shuffle(result)
    return result


def prompt_for(review, examples=()):
    lines = [
        "Classify the sentiment of this movie-review phrase using exactly five levels.",
        "0 = very negative; 1 = somewhat negative; 2 = neutral or mixed;",
        "3 = somewhat positive; 4 = very positive.",
        "Account for negation, sarcasm and strength of sentiment.",
        "Reply ONLY as valid JSON: {\"sentiment\": <integer from 0 to 4>}."
    ]
    for text, label, *_ in examples:
        lines += [f"Example review: {json.dumps(str(text), ensure_ascii=False)}",
                  f"Example answer: {{\"sentiment\": {int(label)}}}"]
    lines.append("Review: " + json.dumps(str(review), ensure_ascii=False))
    lines.append("Answer:")
    return "\n".join(lines)


def parse_llm_label(output):
    """Reject ambiguous/faulty outputs. Never silently clip to a legal class."""
    text = str(output).strip()
    # A model may wrap JSON in a markdown fence or surround it with prose.
    matches = list(re.finditer(r'\{\s*"sentiment"\s*:\s*(\d+)\s*\}', text))
    if len(matches) != 1:
        return -1
    label = int(matches[0].group(1))
    return label if 0 <= label <= 4 else -1


def save_evaluation(frame, y, pred, fold_ids, output_dir, metadata):
    pred = np.asarray(pred, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if pred.shape != (len(y),) or fold_ids.shape != (len(y),):
        raise ValueError("Prediction/assignment shapes do not match")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    complete = fold_ids >= 0
    scored = pred[complete]
    if not (np.isin(scored, [-1, 0, 1, 2, 3, 4]).all()):
        raise ValueError("Unknown predicted classes")
    truth = y[complete]
    scores = {
        "evaluated_rows": int(complete.sum()),
        "total_training_rows": int(len(y)),
        "accuracy": float(accuracy_score(truth, scored)) if len(truth) else None,
        "macro_f1": float(f1_score(truth, scored, labels=LABELS, average="macro",
                                    zero_division=0)) if len(truth) else None,
        "invalid_predictions": int(np.sum(scored == -1)),
        "recall_by_class": (recall_score(truth, scored, labels=LABELS,
                                         average=None, zero_division=0).tolist()
                            if len(truth) else []),
        "confusion_matrix": (confusion_matrix(truth, scored, labels=LABELS).tolist()
                              if len(truth) else []),
    }
    pd.DataFrame({
        "PhraseId": frame.PhraseId, "outer_fold": fold_ids, "label": y,
        "prediction": pred,
    }).to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({
        "protocol": "train-only stratified OOF; no test.csv accessed",
        "metrics": scores, "config": metadata,
        "warning": "Scored subset is a smoke check if not all five folds were evaluated.",
    }, indent=2, default=str))
    print(f"RESULT evaluated={scores['evaluated_rows']}/{len(y)} "
          f"accuracy={scores['accuracy']} macroF1={scores['macro_f1']} "
          f"invalid={scores['invalid_predictions']}", flush=True)
    return scores


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
