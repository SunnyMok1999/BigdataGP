"""Tests for train-only post-training protocol. No pretrained model downloads."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from posttrain_common import (
    load_training_csv, parse_llm_label, prompt_for, save_evaluation,
    splits, training_fewshots,
)


def test_five_class_training_and_fold_disjointness():
    df, y = load_training_csv(ROOT / "train.csv")
    assert len(df) == 511 and set(y.tolist()) == set(range(5))
    all_valid = []
    for tr, va in splits(y, 5, seed=42):
        assert not set(tr) & set(va)
        assert len(set(y[tr])) == 5 and len(set(y[va])) == 5
        all_valid.extend(va)
    assert sorted(all_valid) == list(range(len(y)))


def test_fewshot_never_uses_validation_labels_or_texts():
    df, y = load_training_csv(ROOT / "train.csv")
    tr, va = splits(y)[0]
    sampled = training_fewshots(df, y, tr, seed=43, per_class=2)
    assert len(sampled) == 10
    assert sorted([label for _, label, _ in sampled]) == [0,0,1,1,2,2,3,3,4,4]
    assert all(idx in set(tr) for _, _, idx in sampled)
    modified = y.copy()
    modified[va] = 4
    assert training_fewshots(df, modified, tr, seed=43, per_class=2) == sampled


@pytest.mark.parametrize(("output", "expected"), [
    ('{"sentiment": 0}', 0),
    (' {"sentiment":4}\n', 4),
    ('some preamble {"sentiment": 2}', 2),
    ('{"sentiment": 5}', -1),
    ('{"sentiment": -1}', -1),
    ('{"label": 2}', -1),
    ('{"sentiment": 1} {"sentiment": 2}', -1),
])
def test_llm_json_parser(output, expected):
    assert parse_llm_label(output) == expected


def test_prompt_has_exact_schema_and_selected_examples():
    text = prompt_for("Not good at all", [("Great film!", 4, 0)])
    assert '"sentiment"' in text and "0 = very negative" in text
    assert 'Example answer: {"sentiment": 4}' in text


def test_partial_cv_is_marked_as_smoke_and_doesnt_score_missing(tmp_path):
    df, y = load_training_csv(ROOT / "train.csv")
    tr, va = splits(y)[0]
    preds = np.full(len(y), -1, dtype=int)
    fold_ids = np.full(len(y), -1, dtype=int)
    preds[va] = y[va]
    fold_ids[va] = 0
    result = save_evaluation(df, y, preds, fold_ids, tmp_path, {"smoke": True})
    assert result["evaluated_rows"] == len(va)
    assert (tmp_path/"metrics.json").is_file()
    assert result["macro_f1"] == 1.0
