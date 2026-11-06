"""Audit small projection students using frozen MiniLM training embeddings.

This is a new, documented experiment, not an exact reproduction of unavailable
ChatGPT training code. No test partition is read. Run with the local dml Python:

    python experiments/student_distillation.py

The --predict-features option accepts a new matrix of 384-dimensional embeddings
from the same MiniLM encoder and preprocessing used to create the input features.
"""

from __future__ import annotations

import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/movie-review-student-matplotlib")

import argparse
import hashlib
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             mean_absolute_error, silhouette_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


SEED = 42
LABELS = np.arange(5)
DIM = 16
TEMPERATURE = 0.2
MAX_EPOCHS = 150
PATIENCE = 15
VARIANTS = {
    "ce_only": {"distill": False, "balanced": False, "ordinal": False},
    "teacher_ce": {"distill": True, "balanced": False, "ordinal": False},
    "teacher_ce_balanced": {"distill": True, "balanced": True, "ordinal": False},
    "ordinal_ce": {"distill": False, "balanced": False, "ordinal": True},
}


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, pred, probability=None):
    value = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, labels=LABELS, weights="quadratic")),
        "mean_absolute_error": float(mean_absolute_error(y, pred)),
        "class_0_recall": float(np.mean(np.asarray(pred)[np.asarray(y) == 0] == 0)),
    }
    if probability is not None:
        value["log_loss"] = float(log_loss(y, probability, labels=LABELS))
    return value


def geometry(x, y):
    return {
        "silhouette_euclidean": float(silhouette_score(x, y, metric="euclidean")),
        "silhouette_cosine": float(silhouette_score(x, y, metric="cosine")),
    }


class Student(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, DIM),
        )
        # Fixed orthogonal classifier anchors keep output coordinates comparable
        # across folds. The representation is normalized before prediction.
        self.register_buffer("prototypes", torch.eye(DIM)[:5])

    def forward(self, x):
        embedding = F.normalize(self.network(x), p=2, dim=1)
        return embedding, embedding @ self.prototypes.T / TEMPERATURE


def make_teacher(x, y, residual_projection):
    residual = x @ residual_projection
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    target = np.eye(DIM, dtype=np.float32)[y] + 0.1 * residual
    return target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)


def fit(x, y, target, settings, seed, epochs, validation=None):
    torch.manual_seed(seed)
    model = Student(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    inputs = torch.from_numpy(np.asarray(x, dtype=np.float32))
    labels = torch.from_numpy(np.asarray(y, dtype=np.int64))
    targets = torch.from_numpy(np.asarray(target, dtype=np.float32))
    weights = None
    if settings["balanced"]:
        counts = np.bincount(y, minlength=5)
        weights = torch.tensor(len(y) / (5 * counts), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_epoch, best_nll, since_improved = 1, float("inf"), 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(y), generator=generator)
        total = 0.0
        for batch in permutation.split(64):
            optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(inputs[batch])
            loss = F.cross_entropy(logits, labels[batch], weight=weights)
            if settings["distill"]:
                # Squared Euclidean distance, rather than componentwise MSE,
                # gives the distillation term a scale comparable to CE.
                loss = loss + (embedding - targets[batch]).square().sum(1).mean()
            if settings["ordinal"]:
                expected = logits.softmax(1) @ torch.arange(5, dtype=torch.float32)
                loss = loss + 0.1 * F.smooth_l1_loss(expected, labels[batch].float())
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        row = {"epoch": epoch, "training_loss": total / len(y)}
        if validation is not None:
            val_x, val_y = validation
            model.eval()
            with torch.no_grad():
                _, logits = model(torch.from_numpy(np.asarray(val_x, dtype=np.float32)))
                val_nll = float(F.cross_entropy(logits, torch.from_numpy(val_y.astype(np.int64))))
                val_pred = logits.argmax(1).numpy()
            row.update(validation_nll=val_nll,
                       validation_macro_f1=float(f1_score(val_y, val_pred, labels=LABELS, average="macro", zero_division=0)))
            if val_nll < best_nll - 1e-6:
                best_epoch, best_nll, since_improved = epoch, val_nll, 0
            else:
                since_improved += 1
        history.append(row)
        if validation is not None and since_improved >= PATIENCE:
            break
    return model, best_epoch if validation is not None else epochs, history


def predict(model, x):
    model.eval()
    with torch.no_grad():
        emb, logits = model(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    return emb.numpy(), logits.softmax(1).numpy()


def checkpoint(path, model, scaler, residual_projection, metadata):
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": len(scaler.mean_),
        "scaler_mean": torch.tensor(scaler.mean_),
        "scaler_scale": torch.tensor(scaler.scale_),
        "teacher_residual_projection": torch.from_numpy(residual_projection),
        "metadata": metadata,
    }, path)


def plot_spaces(path, x, teacher, embeddings, y):
    spaces = [("Frozen MiniLM", x), ("Teacher: true labels injected", teacher)]
    spaces += [(f"OOF {name}", value) for name, value in embeddings.items()]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    palette = plt.get_cmap("RdYlGn", 5)
    for axis, (title, space) in zip(axes.flat, spaces):
        reduced = PCA(n_components=2, random_state=SEED).fit_transform(space)
        for label in LABELS:
            take = y == label
            axis.scatter(reduced[take, 0], reduced[take, 1], s=14, alpha=.7,
                         color=palette(int(label)), label=str(label))
        axis.set(title=title, xlabel="PC1 (fit within this panel)", ylabel="PC2")
    axes.flat[0].legend(title="Sentiment", fontsize=8)
    figure.suptitle("Training-label geometry audit: separate PCA per panel; OOF students use different fold fits", fontsize=11)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args):
    started = time.monotonic()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.train)
    rows = pd.read_csv(args.feature_rows)
    x = np.load(args.features, allow_pickle=False).astype(np.float32)
    y = frame.Sentiment.to_numpy(dtype=np.int64)
    if x.ndim != 2 or x.shape != (len(frame), 384):
        raise ValueError(f"Expected {len(frame)} x 384 MiniLM embeddings, got {x.shape}")
    if not np.isfinite(x).all() or set(y) != set(LABELS):
        raise ValueError("Nonfinite features or unexpected sentiment classes")
    for column in ("PhraseId", "SentenceId", "Sentiment"):
        if not np.array_equal(rows[column].to_numpy(), frame[column].to_numpy()):
            raise ValueError(f"Feature and label rows do not align on {column}")
    if frame.SentenceId.duplicated().any() or frame.Phrase.duplicated().any():
        raise ValueError("Duplicate texts or sentence groups require grouped CV")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    residual_projection = np.random.default_rng(SEED).normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], DIM)).astype(np.float32)
    teacher = make_teacher(x, y, residual_projection)
    np.save(out / "label_injected_teacher_embeddings.npy", teacher)
    pd.concat([rows, pd.DataFrame(teacher, columns=[f"embedding_{k}" for k in range(DIM)])], axis=1).to_csv(
        out / "label_injected_teacher_embeddings.csv", index=False)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y))
    folds = np.full(len(y), -1, dtype=np.int64)
    for fold, (_, holdout) in enumerate(splits):
        folds[holdout] = fold
    rows.assign(cv_fold=folds).to_csv(out / "cv_folds.csv", index=False)
    summaries = {}
    oof_embeddings = {}
    selected_epochs = {}
    for variant, settings in VARIANTS.items():
        variant_dir = out / variant
        variant_dir.mkdir(exist_ok=True)
        oof_emb = np.full((len(y), DIM), np.nan, dtype=np.float32)
        oof_proba = np.full((len(y), 5), np.nan, dtype=np.float32)
        fold_rows = []
        histories = []
        epochs = []
        for fold, (training, holdout) in enumerate(splits):
            core, validation = train_test_split(training, test_size=0.2, stratify=y[training], random_state=SEED + fold)
            assert not set(training).intersection(holdout)
            assert not set(core).intersection(validation)
            scaler_inner = StandardScaler().fit(x[core])
            _, best_epoch, history = fit(
                scaler_inner.transform(x[core]), y[core], teacher[core], settings,
                SEED + fold, MAX_EPOCHS,
                validation=(scaler_inner.transform(x[validation]), y[validation]),
            )
            epochs.append(best_epoch)
            for row in history:
                histories.append({"fold": fold, **row})
            scaler_outer = StandardScaler().fit(x[training])
            model, _, _ = fit(scaler_outer.transform(x[training]), y[training], teacher[training],
                              settings, SEED + fold, best_epoch)
            embeddings, probability = predict(model, scaler_outer.transform(x[holdout]))
            oof_emb[holdout] = embeddings
            oof_proba[holdout] = probability
            row = {"fold": fold, "selected_epochs": best_epoch, "training_n": len(training),
                   "inner_validation_n": len(validation), "holdout_n": len(holdout),
                   **metrics(y[holdout], probability.argmax(1), probability),
                   **geometry(embeddings, y[holdout])}
            fold_rows.append(row)
            checkpoint(variant_dir / f"fold_{fold}.pt", model, scaler_outer, residual_projection,
                       {"variant": variant, "fold": fold, "epochs": best_epoch,
                        "training_rows": training.tolist(), "heldout_rows": holdout.tolist()})
            print(f"{variant} fold {fold}: epochs={best_epoch}, accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}", flush=True)
        if not np.isfinite(oof_proba).all() or not np.isfinite(oof_emb).all():
            raise RuntimeError("Missing out-of-fold predictions")
        pred = oof_proba.argmax(1)
        output_rows = rows.assign(cv_fold=folds, predicted_sentiment=pred)
        for label in LABELS:
            output_rows[f"probability_{label}"] = oof_proba[:, label]
        output_rows.to_csv(variant_dir / "oof_predictions.csv", index=False)
        np.save(variant_dir / "oof_embeddings.npy", oof_emb)
        pd.concat([rows.assign(cv_fold=folds), pd.DataFrame(oof_emb, columns=[f"embedding_{k}" for k in range(DIM)])], axis=1).to_csv(
            variant_dir / "oof_embeddings.csv", index=False)
        np.save(variant_dir / "oof_probabilities.npy", oof_proba)
        pd.DataFrame(fold_rows).to_csv(variant_dir / "fold_metrics.csv", index=False)
        pd.DataFrame(histories).to_csv(variant_dir / "inner_early_stopping_history.csv", index=False)
        dump_json(variant_dir / "classification_report.json", classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0))
        pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(variant_dir / "confusion_matrix.csv")
        # A deployment artifact fitted on all 511 training rows. This fit is not
        # an independent evaluation; OOF scores above come from outer-fold fits.
        final_epochs = max(1, int(np.median(epochs)))
        scaler_all = StandardScaler().fit(x)
        model_all, _, _ = fit(scaler_all.transform(x), y, teacher, settings, SEED, final_epochs)
        full_emb, full_proba = predict(model_all, scaler_all.transform(x))
        checkpoint(variant_dir / "full_training_model.pt", model_all, scaler_all, residual_projection,
                   {"variant": variant, "epochs": final_epochs, "epoch_rule": "median outer-fold inner-validation selected epochs",
                    "training_feature_sha256": sha256(args.features), "training_csv_sha256": sha256(args.train),
                    "label_injected_teacher": settings["distill"]})
        summary = {
            "settings": settings, "oof_metrics": metrics(y, pred, oof_proba),
            "fold_mean_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
            "fold_mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
            "fold_sd_accuracy": float(np.std([row["accuracy"] for row in fold_rows], ddof=1)),
            "fold_sd_macro_f1": float(np.std([row["macro_f1"] for row in fold_rows], ddof=1)),
            "oof_geometry_diagnostic": geometry(oof_emb, y),
            "inner_selected_epochs": epochs, "full_fit_epochs": final_epochs,
            "full_fit_training_metrics_not_generalization": metrics(y, full_proba.argmax(1), full_proba),
            "full_fit_geometry_not_generalization": geometry(full_emb, y),
        }
        summaries[variant] = summary
        oof_embeddings[variant] = oof_emb
        selected_epochs[variant] = epochs
        dump_json(variant_dir / "summary.json", summary)
    # Deliberate memorization control: no early stopping and no claimed held-out
    # performance. This answers whether clean training clusters can be learned
    # merely by fitting all known labels, as described in the original chat.
    control_dir = out / "memorization_control"
    control_dir.mkdir(exist_ok=True)
    control_scaler = StandardScaler().fit(x)
    control_model, _, control_history = fit(
        control_scaler.transform(x), y, teacher, VARIANTS["teacher_ce"], SEED, MAX_EPOCHS)
    control_emb, control_probability = predict(control_model, control_scaler.transform(x))
    control_summary = {
        "description": "Deliberately overfit teacher+CE on all training rows for 150 epochs; no independent performance estimate",
        "epochs": MAX_EPOCHS,
        "training_metrics_not_generalization": metrics(y, control_probability.argmax(1), control_probability),
        "training_geometry_not_generalization": geometry(control_emb, y),
    }
    dump_json(control_dir / "summary.json", control_summary)
    pd.DataFrame(control_history).to_csv(control_dir / "training_history.csv", index=False)
    np.save(control_dir / "training_embeddings.npy", control_emb)
    checkpoint(control_dir / "all_training_150_epochs.pt", control_model, control_scaler, residual_projection,
               {"variant": "teacher_ce_memorization_control", "epochs": MAX_EPOCHS,
                "performance_caveat": "Trained and scored on the same 511 examples; no generalization evidence"})
    result = {
        "experiment": "new small projection-student audit, not exact ChatGPT reproduction",
        "sample_count": len(y), "feature_dimensions": x.shape[1],
        "class_counts": {str(k): int((y == k).sum()) for k in LABELS},
        "feature_sha256": sha256(args.features), "training_csv_sha256": sha256(args.train),
        "input_norm_min": float(np.linalg.norm(x, axis=1).min()),
        "input_norm_max": float(np.linalg.norm(x, axis=1).max()),
        "original_frozen_geometry": geometry(x, y),
        "label_injected_teacher_geometry_not_generalization": geometry(teacher, y),
        "protocol": {"outer_folds": 5, "outer_stratified_shuffle_seed": SEED,
                     "inner_validation_fraction": 0.2, "inner_seed": "42 + fold",
                     "scaler": "fit inner-train for early stopping; refit outer-train for final fold fit",
                     "early_stopping": "minimum unweighted inner-validation cross entropy; patience 15; max 150 epochs",
                     "architecture": "384 -> 64 ReLU Dropout(0.3) -> 16, unit-normalized; fixed orthogonal prototypes",
                     "optimizer": "AdamW(lr=.001, weight_decay=.01), batch_size=64",
                     "teacher": "normalize(orthogonal_sentiment_center + .1 * normalized fixed Gaussian projection of raw MiniLM input)",
                     "teacher_loss": "CE + squared Euclidean distance to teacher target",
                     "ordinal_loss": "CE + .1 * SmoothL1(expected sentiment, true label)",
                     "class_balance": "training-fold inverse frequency weights for CE only",
                     "test_partition_accessed": False,
                     "geometry_caveat": "OOF points come from different fitted students; fixed anchors reduce arbitrary rotations but do not make cross-fold geometry equivalent to a single unseen-data encoder. Per-fold silhouette is also saved.",
                     "comparison_caveat": "Four prespecified variants were evaluated. Selecting the best using these OOF scores adds model-selection optimism. No multiple-comparison significance claim."},
        "variants": summaries,
        "memorization_control": control_summary,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__},
        "elapsed_seconds": time.monotonic() - started,
    }
    dump_json(out / "summary.json", result)
    pd.DataFrame([{"variant": name, **value["oof_metrics"],
                   **value["oof_geometry_diagnostic"], "full_fit_training_accuracy": value["full_fit_training_metrics_not_generalization"]["accuracy"]}
                  for name, value in summaries.items()]).to_csv(out / "comparison.csv", index=False)
    plot_spaces(out / "embedding_geometry_audit.png", x, teacher, oof_embeddings, y)
    (out / "README.md").write_text("""# Projection-student audit

This is a new implementation inspired by the supplied ChatGPT description; its
unavailable code, targets, splits, and hyperparameters cannot be reproduced
exactly. Only train.csv and frozen MiniLM features in that same row order are
read. The original test partition is never opened.

Each outer holdout is excluded from scaling, early stopping, fitting, and
teacher targets used for training. A stratified 20% inner validation selects an
epoch by cross entropy; the student is then fitted from scratch on the entire
outer training fold for exactly that many epochs. The four variants and all
training settings are specified in student_distillation.py and summary.json.

Teacher targets explicitly inject true sentiment labels; their silhouette is a
construction diagnostic, not evidence of generalization. Out-of-fold embeddings
come from separate fitted students. Shared fixed prototype coordinates prevent
arbitrary rotations, but pooled OOF silhouette is still a diagnostic; consult
per-fold silhouettes too. PCA panels have separate axes and do not establish
class separability beyond the quantitative metrics.

The full_training_model.pt files use all 511 training rows and median selected
epochs. Their in-sample training accuracy is not an unbiased performance estimate.
The separate memorization_control directory deliberately trains teacher+CE on
all rows for 150 epochs. Its scores and geometry use those same training rows;
they demonstrate memorization capacity and cannot establish generalization.
The OOF metrics estimate each prespecified training procedure; choosing the best
of these variants after seeing the scores can be optimistic. No significance or
independent test-performance claim is made. Baselines and encoder experiments
must be compared on identical rows, labels, and held-out folds.

For inference on new MiniLM embeddings (same encoder and text preprocessing):

    python experiments/student_distillation.py --predict-features new_minilm.npy \\
        --checkpoint experiments/results/student/teacher_ce/full_training_model.pt \\
        --prediction-output predictions.npz

The prediction archive contains embeddings, probabilities and predicted labels;
no true labels are needed for inference.
""")
    print(pd.read_csv(out / "comparison.csv").to_string(index=False), flush=True)


def infer(args):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required with --predict-features")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    x = np.load(args.predict_features, allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != saved["n_features"] or not np.isfinite(x).all():
        raise ValueError("Unexpected inference feature shape or nonfinite values")
    transformed = (x - saved["scaler_mean"].numpy()) / saved["scaler_scale"].numpy()
    model = Student(saved["n_features"])
    model.load_state_dict(saved["state_dict"])
    emb, proba = predict(model, transformed)
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.prediction_output, embeddings=emb, probabilities=proba,
                        predicted_sentiment=proba.argmax(1))
    print(f"Saved {len(x)} predictions to {args.prediction_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--features", type=Path, default=Path("experiments/features/minilm_train.npy"))
    parser.add_argument("--feature-rows", type=Path, default=Path("experiments/features/minilm_train_rows.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/student"))
    parser.add_argument("--predict-features", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prediction-output", type=Path, default=Path("student_predictions.npz"))
    arguments = parser.parse_args()
    infer(arguments) if arguments.predict_features else run(arguments)
