"""CPU integrity and scientific boundary checks for the local experiments.

Run with the dml Python. This file never reads the supplied test partition or
loads pretrained weights. Missing experiment outputs are reported as pending;
available artifacts are checked against the original training IDs and labels.
The final stdout JSON is suitable for saving as an experiment audit artifact.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
CHECKS = []


def check(name, fn):
    try:
        detail = fn()
        CHECKS.append({"check": name, "status": "passed", "detail": detail})
    except Exception as error:
        CHECKS.append({"check": name, "status": "failed",
                       "detail": f"{type(error).__name__}: {error}"})


def pending(name, detail):
    CHECKS.append({"check": name, "status": "pending", "detail": detail})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_rows(rows, frame):
    assert len(rows) == len(frame), "Wrong number of training rows"
    for column in ["PhraseId", "SentenceId", "Sentiment"]:
        if column in rows:
            assert np.array_equal(rows[column].to_numpy(), frame[column].to_numpy()), column
    if "row_index" in rows:
        assert np.array_equal(rows.row_index, np.arange(len(frame))), "Row order changed"
    assert rows.PhraseId.is_unique, "OOF row duplicated"


def unit_vectors(values, count):
    assert values.ndim == 2 and len(values) == count
    assert np.isfinite(values).all()
    assert np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-5)


def probabilities(values, count):
    assert values.shape == (count, 5), values.shape
    assert np.isfinite(values).all()
    assert (values >= 0).all() and (values <= 1).all()
    assert np.allclose(values.sum(1), 1., atol=1e-5)


def partition(train_rows, val_rows, expected, labels):
    tr, va = np.asarray(train_rows), np.asarray(val_rows)
    assert len(tr) == len(np.unique(tr)) and len(va) == len(np.unique(va))
    assert not set(tr) & set(va), "Training/validation overlap"
    assert set(tr) | set(va) == set(expected), "Incomplete partition"
    assert set(labels[tr]) == set(range(5)) and set(labels[va]) == set(range(5))


def scientific_losses(encoder_module):
    labels = torch.tensor([0, 0, 4, 4])
    separated = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]])
    collapsed = F.normalize(torch.ones(4, 2), dim=1)
    for fn in [encoder_module.supervised_contrastive, encoder_module.ordinal_hard_negative]:
        low, high = fn(separated, labels), fn(collapsed, labels)
        assert torch.isfinite(low) and torch.isfinite(high) and low < high
    # With all pair similarities equal there are exactly three denominator terms.
    collapsed_loss = encoder_module.supervised_contrastive(collapsed, labels)
    assert torch.allclose(collapsed_loss, torch.tensor(np.log(3), dtype=torch.float32), atol=1e-6)
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(encoder_module.supervised_contrastive(separated, labels),
                          encoder_module.supervised_contrastive(separated[perm], labels[perm]), atol=1e-6)
    # The ordinal margin must penalize a 0-versus-4 collision more than 0-versus-1.
    distant = encoder_module.ordinal_hard_negative(collapsed, labels)
    adjacent = encoder_module.ordinal_hard_negative(collapsed, torch.tensor([0, 0, 1, 1]))
    assert distant > adjacent > 0
    for fn in [encoder_module.supervised_contrastive, encoder_module.ordinal_hard_negative]:
        for boundary_labels in [torch.zeros(4, dtype=torch.long), torch.arange(4)]:
            z = F.normalize(torch.randn(4, 5), dim=1).detach().requires_grad_()
            value = fn(z, boundary_labels)
            value.backward()
            assert torch.isfinite(value) and torch.isfinite(z.grad).all()
            # SupCon has no positive pairs with unique labels. Hard-negative loss
            # has no usable pair set for either unique labels or a single class.
            if fn is encoder_module.ordinal_hard_negative or len(boundary_labels.unique()) == 4:
                assert value.item() == 0
    return "Separated labels improve both losses; ordinal distance, permutation, no-positive and no-negative cases pass"


def auxiliary_gradients(encoder_module):
    class SyntheticHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(6, 5)
            self.projector = nn.Linear(6, 16)
            # Even if target-side tensors could require gradients, targets must
            # be detached by the auxiliary loss, not by the test's input setup.
            self.anchors = torch.eye(16)[:5].requires_grad_()
            self.semantic_projection = torch.randn(6, 16).requires_grad_()

    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    gradient_norms = {}
    for objective in ["supcon", "teacher", "ordinal", "hard_negative"]:
        torch.manual_seed(91)
        model = SyntheticHeads()
        latent = torch.randn(10, 6, requires_grad=True)
        teacher = torch.randn(10, 6, requires_grad=True)
        embedding = F.normalize(latent, dim=1)
        logits = model.classifier(embedding) * 10.
        projected = F.normalize(model.projector(embedding), dim=1)
        total, parts = encoder_module.loss_parts(model, logits, embedding, projected,
                                                  labels, teacher, {objective: 1.})
        # Remove CE analytically to test the auxiliary signal independently.
        auxiliary = total - F.cross_entropy(logits, labels)
        auxiliary.backward()
        assert torch.isfinite(total) and all(np.isfinite(x) for x in parts.values())
        assert latent.grad is not None and torch.isfinite(latent.grad).all()
        assert latent.grad.norm() > 0, f"No encoder signal from {objective}"
        assert teacher.grad is None, "Teacher features received gradients"
        assert model.anchors.grad is None and model.semantic_projection.grad is None
        if objective == "teacher":
            assert model.projector.weight.grad is not None
            assert model.projector.weight.grad.norm() > 0
        gradient_norms[objective] = float(latent.grad.norm())
    return {"finite_nonzero_encoder_gradient_norms": gradient_norms,
            "target_side_gradients": "none"}


def main():
    torch.set_num_threads(1)
    frame = pd.read_csv(ROOT / "train.csv")
    y = frame.Sentiment.to_numpy()
    n = len(frame)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(frame, y))
    expected_folds = np.full(n, -1)
    for fold, (_, va) in enumerate(splits):
        expected_folds[va] = fold

    def training_integrity():
        assert n == 511 and set(y) == set(range(5))
        assert frame[["PhraseId", "SentenceId", "Phrase", "Sentiment"]].notna().all().all()
        assert frame.PhraseId.is_unique and frame.SentenceId.is_unique and frame.Phrase.is_unique
        return {"rows": n, "class_counts": frame.Sentiment.value_counts().sort_index().to_dict()}
    check("training_data_integrity", training_integrity)

    def fold_integrity():
        seen = np.zeros(n, dtype=int)
        sizes = []
        for fold, (tr, va) in enumerate(splits):
            partition(tr, va, np.arange(n), y)
            inner_tr, inner_va = train_test_split(tr, test_size=.2, stratify=y[tr], random_state=42 + fold)
            partition(inner_tr, inner_va, tr, y)
            assert not set(va) & (set(inner_tr) | set(inner_va))
            seen[va] += 1
            sizes.append(dict(outer_train=len(tr), outer_val=len(va),
                              inner_train=len(inner_tr), inner_val=len(inner_va)))
        assert (seen == 1).all()
        return sizes
    check("five_outer_folds_and_inner_holdouts_disjoint_and_all_classes", fold_integrity)

    feature_paths = sorted((ROOT / "experiments/features").glob("*_train_metadata.json"))
    if not feature_paths:
        pending("frozen_features", "No completed cached embedding artifacts yet")
    for metadata_path in feature_paths:
        key = metadata_path.name.removesuffix("_train_metadata.json")
        def feature_integrity(key=key, metadata_path=metadata_path):
            directory = metadata_path.parent
            meta = json.loads(metadata_path.read_text())
            array_path = directory / f"{key}_train.npy"
            array = np.load(array_path)
            rows = pd.read_csv(directory / f"{key}_train_rows.csv")
            same_rows(rows, frame)
            unit_vectors(array, n)
            assert meta["samples"] == n and meta["dimensions"] == array.shape[1]
            assert meta["source_file_sha256"] == digest(ROOT / "train.csv")
            assert meta["embedding_file_sha256"] == digest(array_path)
            assert meta["text_sha256"] == hashlib.sha256("\n".join(frame.Phrase).encode("utf8")).hexdigest()
            return {"shape": list(array.shape), "row_alignment": True, "hashes_match": True}
        check(f"feature_integrity_{key}", feature_integrity)

    spec = importlib.util.spec_from_file_location("local_encoder", ROOT / "experiments/encoder_finetuning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check("scientific_loss_geometry_and_boundary_cases", lambda: scientific_losses(module))
    check("independent_auxiliary_gradients_and_teacher_detachment", lambda: auxiliary_gradients(module))

    encoder_folds = sorted((RESULTS / "encoder").glob("*/fold_[0-4].json"))
    if not encoder_folds:
        pending("saved_encoder_folds", "No completed encoder fold artifact yet")
    for path in encoder_folds:
        def encoder_fold_integrity(path=path):
            meta = json.loads(path.read_text())
            fold = int(meta["fold"])
            tr, va = splits[fold]
            assert meta["outer_train_rows"] == tr.tolist() and meta["outer_val_rows"] == va.tolist()
            partition(meta["inner_train_rows"], meta["inner_val_rows"], tr, y)
            assert not set(meta["inner_train_rows"]) & set(va)
            arrays = np.load(path.with_suffix(".npz"))
            assert np.array_equal(arrays["rows"], va)
            probabilities(arrays["probabilities"], len(va))
            unit_vectors(arrays["embeddings"], len(va))
            unit_vectors(arrays["projections"], len(va))
            score = f1_score(y[va], arrays["probabilities"].argmax(1), average="macro", zero_division=0)
            assert np.isclose(score, meta["macro_f1"])
            assert meta["selected_epochs"] >= 1
            history = meta["inner_history"]
            assert int(meta["selected_epochs"]) == min(history, key=lambda h: h["val_nll"])["epoch"]
            return {"validation_rows": len(va), "metrics_recomputed": True}
        check(f"encoder_fold_{path.parent.name}_{path.stem}", encoder_fold_integrity)

    frozen_metrics = sorted((RESULTS / "frozen").glob("*/metrics.json"))
    if not frozen_metrics:
        pending("frozen_nested_oof", "No completed frozen classifier artifact yet")
    for path in frozen_metrics:
        def frozen_integrity(path=path):
            rows = pd.read_csv(path.parent / "nested_oof_predictions.csv")
            meta = json.loads(path.read_text())
            same_rows(rows, frame)
            assert np.array_equal(rows.outer_fold, expected_folds)
            array = np.load(path.parent / "nested_oof_probabilities.npy")
            probabilities(array, n)
            assert np.allclose(array, rows[[f"prob_{i}" for i in range(5)]])
            assert np.array_equal(rows.prediction, array.argmax(1))
            assert np.isclose(meta["nested_oof"]["macro_f1"], f1_score(y, array.argmax(1), average="macro"))
            seen = np.zeros(n, dtype=int)
            for fold_meta in meta["folds"]:
                fold = fold_meta["fold"]
                tr, va = splits[fold]
                assert fold_meta["train_indices"] == tr.tolist() and fold_meta["validation_indices"] == va.tolist()
                seen[va] += 1
                # Reconstruct inner folds and verify no outer-validation member
                # can enter the three-fold regularization search.
                for inner_tr, inner_va in StratifiedKFold(3, shuffle=True, random_state=42 + fold).split(tr, y[tr]):
                    partition(tr[inner_tr], tr[inner_va], tr, y)
                    assert not set(va) & (set(tr[inner_tr]) | set(tr[inner_va]))
            assert (seen == 1).all()
            return {"oof_rows": n, "nested_boundaries": True, "metrics_recomputed": True}
        check(f"frozen_nested_oof_{path.parent.name}", frozen_integrity)

    baseline_dir = RESULTS / "baseline"
    if (baseline_dir / "selected_configuration.json").exists():
        def baseline_integrity():
            rows = pd.read_csv(baseline_dir / "selected_model_oof_predictions.csv")
            same_rows(rows, frame)
            assert np.array_equal(rows.cv_fold, expected_folds)
            candidates = pd.read_csv(baseline_dir / "cv_candidates.csv")
            assert len(candidates) == 48
            assert candidates[[f"split{i}_test_macro_f1" for i in range(5)]].notna().all().all()
            meta = json.loads((baseline_dir / "selected_configuration.json").read_text())
            assert "optimistic" in meta["selection_bias_warning"]
            assert np.isclose(meta["oof_metrics"]["macro_f1"], f1_score(y, rows.predicted_sentiment, average="macro"))
            return {"candidates": len(candidates), "oof_rows": n, "selection_bias_disclosed": True}
        check("baseline_artifacts", baseline_integrity)
    else:
        pending("baseline_artifacts", "No completed baseline artifact yet")

    student_paths = sorted((RESULTS / "student").glob("*/summary.json"))
    if not student_paths:
        pending("student_oof_artifacts", "No completed student variant yet")
    for path in student_paths:
        def student_integrity(path=path):
            rows = pd.read_csv(path.parent / "oof_predictions.csv")
            same_rows(rows, frame)
            assert np.array_equal(rows.cv_fold, expected_folds)
            array = np.load(path.parent / "oof_probabilities.npy")
            probabilities(array, n)
            assert np.allclose(array, rows[[f"probability_{i}" for i in range(5)]])
            assert np.array_equal(rows.predicted_sentiment, array.argmax(1))
            unit_vectors(np.load(path.parent / "oof_embeddings.npy"), n)
            return {"oof_rows": n, "probabilities_and_embeddings": "finite and normalized"}
        check(f"student_oof_{path.parent.name}", student_integrity)

    encoder_summaries = sorted((RESULTS / "encoder").glob("*/summary.json"))
    if not encoder_summaries:
        pending("encoder_complete_oof", "No completed five-fold encoder configuration yet")
    for path in encoder_summaries:
        def encoder_oof_integrity(path=path):
            meta = json.loads(path.read_text())
            rows = pd.read_csv(path.parent / "oof_predictions.csv")
            same_rows(rows, frame)
            array = rows[[f"prob_{i}" for i in range(5)]].to_numpy()
            probabilities(array, n)
            assert np.array_equal(rows.prediction, array.argmax(1))
            assert np.isclose(meta["macro_f1"], f1_score(y, array.argmax(1), average="macro"))
            assert np.isclose(meta["accuracy"], accuracy_score(y, array.argmax(1)))
            unit_vectors(np.load(path.parent / "oof_embeddings.npy"), n)
            seen = np.zeros(n, dtype=int)
            for fold in range(5):
                arrays = np.load(path.parent / f"fold_{fold}.npz")
                indices = arrays["rows"]
                assert np.array_equal(indices, splits[fold][1])
                assert np.allclose(array[indices], arrays["probabilities"], atol=1e-7)
                seen[indices] += 1
            assert (seen == 1).all()
            return {"oof_rows": n, "validation_appearances_per_row": 1, "metrics_recomputed": True}
        check(f"encoder_complete_oof_{path.parent.name}", encoder_oof_integrity)

    counts = {status: sum(row["status"] == status for row in CHECKS)
              for status in ["passed", "failed", "pending"]}
    report = {"device": "cpu", "test_partition_read": False,
              "pretrained_weights_loaded": False, "counts": counts, "checks": CHECKS}
    print(json.dumps(report, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
