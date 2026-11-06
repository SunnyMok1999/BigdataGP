#!/usr/bin/env python3
"""Freeze training-only model selection, then perform one prespecified test batch.

    python experiments/select_and_evaluate.py select
    python experiments/encoder_finetuning.py final --configs <locked encoder> --device mps
    python experiments/select_and_evaluate.py evaluate --device mps

The select command never opens a test partition. Existing selection locks are
immutable and source hashes must still match. evaluate computes every comparator
before opening test labels, and never changes selection from test results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
for _name, _value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
                      ("USE_TF", "0"), ("USE_FLAX", "0"),
                      ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_name, _value)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/movie-review-final-matplotlib")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, recall_score)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
SELECTION = RESULTS / "selection"
FINAL = RESULTS / "final"
LOCK = SELECTION / "selection_lock.json"
LABELS = np.arange(5)
FROZEN = ["minilm", "bge_small", "bert_base", "distilbert"]
STUDENTS = ["ce_only", "teacher_ce", "teacher_ce_balanced", "ordinal_ce"]
ENCODERS = ["minilm_ce", "minilm_ce_low_lr", "minilm_last2_ce", "minilm_supcon",
            "minilm_teacher", "minilm_ordinal", "minilm_hard_negative",
            "minilm_combined", "bge_ce"]
COMPARATORS = ["minilm_frozen", "selected_frozen", "selected_student", "tfidf",
               "selected_encoder", "selected_blend"]
N_BOOTSTRAP = 2000


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False,
                  default=lambda item: item.item() if isinstance(item, np.generic) else str(item))
        handle.write("\n")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def valid_probabilities(value, n):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n, 5) or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"Invalid five-class probabilities, shape={value.shape}")
    if not np.allclose(value.sum(1), 1, atol=1e-5):
        raise ValueError("Probability rows do not sum to one")
    return value / value.sum(1, keepdims=True)


def probability_metrics(y, probabilities):
    probabilities = valid_probabilities(probabilities, len(y))
    pred = probabilities.argmax(1)
    result = prediction_metrics(y, pred)
    confidence = probabilities.max(1)
    correct = (pred == y).astype(float)
    ece = 0.
    bins = []
    for i in range(10):
        take = ((confidence >= i / 10) if i == 0 else (confidence > i / 10)) & (confidence <= (i + 1) / 10)
        n = int(take.sum())
        row = {"lower": i / 10, "upper": (i + 1) / 10, "count": n}
        if n:
            average_confidence, average_accuracy = float(confidence[take].mean()), float(correct[take].mean())
            ece += n / len(y) * abs(average_confidence - average_accuracy)
            row.update(mean_confidence=average_confidence, accuracy=average_accuracy)
        bins.append(row)
    result.update(nll=float(log_loss(y, probabilities, labels=LABELS)), ece_10_bins=float(ece),
                  brier_multiclass_sum=float(np.mean(np.sum((probabilities - np.eye(5)[y]) ** 2, axis=1))),
                  calibration_bins=bins)
    return result


def prediction_metrics(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
            "class_0_recall": float(recall_score(y, pred, labels=[0], average=None, zero_division=0)[0]),
            "ordinal_mae": float(np.mean(np.abs(y - pred))),
            "within_one_accuracy": float(np.mean(np.abs(y - pred) <= 1)),
            "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, labels=LABELS, weights="quadratic")),
            "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
            "classification_report": classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)}


def scalar_metrics(value):
    return {name: x for name, x in value.items() if np.isscalar(x)}


def scaled(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1.)) / temperature
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_temperature(probabilities, y):
    fit = minimize_scalar(lambda temperature: log_loss(y, scaled(probabilities, temperature), labels=LABELS),
                          method="bounded", bounds=(.25, 4.), options={"xatol": 1e-6})
    # The explicit boundaries and T=1 avoid excluding a minimum at an endpoint.
    candidates = [.25, 1., 4., float(fit.x)]
    temperature = min(candidates, key=lambda t: log_loss(y, scaled(probabilities, t), labels=LABELS))
    return float(temperature)


def read_oof(path, train, prefix):
    frame = pd.read_csv(path)
    if len(frame) != len(train) or not np.array_equal(frame.PhraseId.to_numpy(), train.PhraseId.to_numpy()):
        raise ValueError(f"OOF IDs/order mismatch: {path}")
    if "Sentiment" in frame and not np.array_equal(frame.Sentiment.to_numpy(), train.Sentiment.to_numpy()):
        raise ValueError(f"OOF labels mismatch: {path}")
    return valid_probabilities(frame[[f"{prefix}{label}" for label in LABELS]].to_numpy(), len(train))


def validate_lock():
    if not LOCK.exists():
        raise FileNotFoundError("Select training procedures and save selection_lock.json before evaluation")
    digest = sha256(LOCK)
    if (SELECTION / "selection_lock.sha256").read_text().strip() != digest:
        raise ValueError("Selection lock integrity mismatch")
    value = json.loads(LOCK.read_text())
    for name, expected in value["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Source changed after immutable selection: {name}")
    return value, digest


def select():
    if LOCK.exists():
        lock, digest = validate_lock()
        print(json.dumps({"existing_immutable_selection_lock_sha256": digest, "selected": lock["selected"],
                          "primary": lock["primary"]}, indent=2))
        return
    SELECTION.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    y = train.Sentiment.to_numpy(dtype=int)
    sources = {ROOT / "train.csv", ROOT / "Cleaned data/cleaned_train.csv", ROOT / "experiments/PROTOCOL.md"}
    records, oof = [], {}

    def add(family, name, path, summary, prefix):
        if not path.exists() or not summary.exists():
            raise FileNotFoundError(f"All prespecified configurations must finish before selection: {path}")
        probs = read_oof(path, train, prefix)
        candidate = f"{family}/{name}"
        value = probability_metrics(y, probs)
        oof[candidate] = probs
        sources.update([path, summary])
        records.append({"candidate": candidate, "family": family, "name": name,
                        **scalar_metrics(value), "selection_optimism": "Family/config selection uses these OOF results"})

    for name in FROZEN:
        folder = RESULTS / "frozen" / name
        add("frozen", name, folder / "nested_oof_predictions.csv", folder / "metrics.json", "prob_")
        sources.add(folder / "classifier.joblib")
        sources.add(ROOT / f"experiments/features/{name}_train_metadata.json")
    for name in STUDENTS:
        folder = RESULTS / "student" / name
        add("student", name, folder / "oof_predictions.csv", folder / "summary.json", "probability_")
        sources.add(folder / "full_training_model.pt")
    for name in ENCODERS:
        folder = RESULTS / "encoder" / name
        add("encoder", name, folder / "oof_predictions.csv", folder / "summary.json", "prob_")

    baseline = RESULTS / "baseline"
    reference = pd.read_csv(baseline / "selected_model_oof_predictions.csv")
    if not np.array_equal(reference.PhraseId.to_numpy(), train.PhraseId.to_numpy()):
        raise ValueError("TFIDF OOF row IDs do not match raw training data")
    if not np.array_equal(reference.Sentiment.to_numpy(), y):
        raise ValueError("TFIDF OOF labels do not match raw training data")
    baseline_metrics = prediction_metrics(y, reference.predicted_sentiment.to_numpy())
    records.append({"candidate": "baseline/tfidf", "family": "baseline", "name": "tfidf",
                    **scalar_metrics(baseline_metrics), "nll": None,
                    "selection_optimism": "Representation and hyperparameters selected on these same folds"})
    sources.update([baseline / name for name in ["selected_configuration.json", "selected_model.joblib", "selected_model_oof_predictions.csv"]])

    def ranking(row):
        return (-row["macro_f1"], float(row["nll"]) if row.get("nll") is not None else float("inf"), row["candidate"])

    best = {family: min([row for row in records if row["family"] == family], key=ranking)
            for family in ["frozen", "student", "encoder"]}
    frozen_probs, encoder_probs = oof[best["frozen"]["candidate"]], oof[best["encoder"]["candidate"]]
    temperatures = {"selected_frozen": fit_temperature(frozen_probs, y),
                    "selected_encoder": fit_temperature(encoder_probs, y)}
    frozen_scaled = scaled(frozen_probs, temperatures["selected_frozen"])
    encoder_scaled = scaled(encoder_probs, temperatures["selected_encoder"])
    blend_grid = []
    for weight in np.linspace(0., 1., 21):
        value = probability_metrics(y, weight * encoder_scaled + (1 - weight) * frozen_scaled)
        blend_grid.append({"encoder_weight": round(float(weight), 2), **scalar_metrics(value)})
    selected_blend = min(blend_grid, key=lambda row: (-row["macro_f1"], row["nll"], row["encoder_weight"]))
    pd.DataFrame(blend_grid).to_csv(SELECTION / "blend_weight_search.csv", index=False)
    weight = selected_blend["encoder_weight"]
    blend_probs = weight * encoder_scaled + (1 - weight) * frozen_scaled
    np.savez_compressed(SELECTION / "selected_calibrated_oof.npz", labels=y,
                        selected_frozen=frozen_scaled, selected_encoder=encoder_scaled, selected_blend=blend_probs)
    calibration = {"warning": "Temperature and blend diagnostics are in-sample with respect to OOF calibration/weight fitting; they are not independent calibration estimates.",
                   "temperatures": temperatures,
                   "selected_frozen": {"raw": probability_metrics(y, frozen_probs), "calibrated": probability_metrics(y, frozen_scaled)},
                   "selected_encoder": {"raw": probability_metrics(y, encoder_probs), "calibrated": probability_metrics(y, encoder_scaled)},
                   "selected_blend": probability_metrics(y, blend_probs)}
    write_json(SELECTION / "oof_calibration_tuning_diagnostics.json", calibration)
    records.append({"candidate": "blend/selected", "family": "blend", "name": "selected",
                    **scalar_metrics(calibration["selected_blend"]),
                    "selection_optimism": "Temperatures and blend weight selected on these OOF predictions"})
    pd.DataFrame(records).sort_values(["macro_f1", "nll"], ascending=[False, True]).to_csv(SELECTION / "comparison.csv", index=False)
    primary_candidates = [dict(best[family], procedure=f"selected_{family}") for family in ["frozen", "student", "encoder"]]
    primary_candidates += [dict(next(row for row in records if row["candidate"] == "baseline/tfidf"), procedure="tfidf"),
                           dict(records[-1], procedure="selected_blend")]
    primary = min(primary_candidates, key=ranking)
    # Pin selection-side outputs too, so deployment and evaluation share one recipe.
    sources.update([SELECTION / "comparison.csv", SELECTION / "blend_weight_search.csv",
                    SELECTION / "selected_calibrated_oof.npz", SELECTION / "oof_calibration_tuning_diagnostics.json"])
    value = {"created_utc": datetime.now(timezone.utc).isoformat(), "training_rows": len(train),
             "selection_metric": "Training out-of-fold five-class macro-F1; ties use NLL then stable candidate name",
             "selected": {family: row["name"] for family, row in best.items()},
             "selected_oof_metrics": {family: {k: row.get(k) for k in ["accuracy", "macro_f1", "class_0_recall", "nll"]}
                                      for family, row in best.items()},
             "primary": {"procedure": primary["procedure"], "candidate": primary["candidate"],
                         "training_oof_macro_f1": primary["macro_f1"]},
             "temperature_scaling": temperatures, "blend": {"encoder_weight": weight, "frozen_weight": 1 - weight,
                                                              "training_oof_macro_f1": selected_blend["macro_f1"]},
             "prespecified_test_comparators": COMPARATORS, "bootstrap": {"replicates": N_BOOTSTRAP, "seed": 42,
                 "scheme": "Paired, independently resample within each true sentiment class; retain original class counts",
                 "differences": ["primary minus minilm_frozen", "primary minus selected_frozen"]},
             "warnings": ["No test partition was opened by select.",
                          "Family/configuration selection adds optimism to OOF performance.",
                          "Temperature and blend fitting uses the OOF labels; its training diagnostics are optimistic.",
                          "The previous project report already contained an older test result; this test is isolated from all new choices."],
             "source_sha256": {relative(path): sha256(path) for path in sorted(sources)}}
    write_json(LOCK, value, exclusive=True)
    (SELECTION / "selection_lock.sha256").write_text(sha256(LOCK) + "\n")
    print(json.dumps({"selection_lock": str(LOCK), "sha256": sha256(LOCK),
                      "selected": value["selected"], "primary": value["primary"],
                      "temperatures": temperatures, "blend": value["blend"]}, indent=2))


def save_predictions(name, frame, probabilities):
    probabilities = valid_probabilities(probabilities, len(frame))
    output = frame[["PhraseId", "SentenceId"]].copy()
    output.insert(0, "row_index", np.arange(len(frame)))
    output["prediction"] = probabilities.argmax(1)
    for label in LABELS:
        output[f"prob_{label}"] = probabilities[:, label]
    output.to_csv(FINAL / f"{name}_predictions.csv", index=False)
    np.save(FINAL / f"{name}_probabilities.npy", probabilities)
    return probabilities


def bootstrap(y, probabilities, primary):
    predictions = {name: probs.argmax(1) for name, probs in probabilities.items()}
    groups = [np.flatnonzero(y == label) for label in LABELS]
    rng = np.random.default_rng(42)
    values = {name: np.empty(N_BOOTSTRAP) for name in predictions}

    def quick_f1(actual, predicted):
        matrix = np.bincount(actual * 5 + predicted, minlength=25).reshape(5, 5)
        denominator = matrix.sum(0) + matrix.sum(1)
        return float(np.mean(np.divide(2 * matrix.diagonal(), denominator,
                                        out=np.zeros(5, dtype=float), where=denominator != 0)))

    for replicate in range(N_BOOTSTRAP):
        rows = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        for name, predicted in predictions.items():
            values[name][replicate] = quick_f1(y[rows], predicted[rows])
    results = {"replicates": N_BOOTSTRAP, "seed": 42, "method": "Paired class-stratified percentile bootstrap",
               "interpretation": "Finite-test-sample uncertainty conditional on fitted models and class counts; excludes training randomness and selection uncertainty.",
               "macro_f1": {}, "primary_differences": {}}
    for name, value in values.items():
        results["macro_f1"][name] = {"estimate": quick_f1(y, predictions[name]),
                                    "lower_95": float(np.quantile(value, .025)), "upper_95": float(np.quantile(value, .975))}
    replicate_frame = pd.DataFrame(values)
    for reference in ["minilm_frozen", "selected_frozen"]:
        difference = values[primary] - values[reference]
        key = f"{primary}_minus_{reference}"
        replicate_frame[key] = difference
        results["primary_differences"][reference] = {"primary": primary,
            "estimate": results["macro_f1"][primary]["estimate"] - results["macro_f1"][reference]["estimate"],
            "lower_95": float(np.quantile(difference, .025)), "upper_95": float(np.quantile(difference, .975))}
    replicate_frame.to_csv(FINAL / "paired_stratified_bootstrap_replicates.csv", index=False)
    write_json(FINAL / "paired_stratified_bootstrap.json", results)
    return results


def make_figures(y, scores, intervals, train_y, original_train, original_test, tuned_train, tuned_test, encoder_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for axis, name in zip(axes.flat, COMPARATORS):
        matrix = np.asarray(scores[name]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        for i in LABELS:
            for j in LABELS:
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                          color="white" if matrix[i, j] > matrix.max() * .55 else "black")
        axis.set(title=name.replace("_", " "), xlabel="Predicted sentiment", ylabel="True sentiment", xticks=LABELS, yticks=LABELS)
    figure.suptitle("Prespecified final test comparisons (341 reviews)")
    figure.savefig(FINAL / "test_confusion_matrices.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    estimates = [scores[name]["macro_f1"] for name in COMPARATORS]
    lower = [max(0, scores[name]["macro_f1"] - intervals["macro_f1"][name]["lower_95"]) for name in COMPARATORS]
    upper = [max(0, intervals["macro_f1"][name]["upper_95"] - scores[name]["macro_f1"]) for name in COMPARATORS]
    axis.bar(np.arange(len(COMPARATORS)), estimates, color="#277da1", yerr=[lower, upper], capsize=4)
    axis.set(xticks=np.arange(len(COMPARATORS)), xticklabels=[name.replace("_", "\n") for name in COMPARATORS],
             ylabel="Macro-F1", ylim=(0, min(1, max(np.asarray(estimates) + upper) + .12)),
             title="Final test macro-F1 with paired class-stratified bootstrap 95% intervals")
    for i, estimate in enumerate(estimates):
        axis.text(i, estimate + upper[i] + .015, f"{estimate:.3f}", ha="center", fontsize=10)
    figure.savefig(FINAL / "test_comparison.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    palette = plt.get_cmap("RdYlGn", 5)
    for column, (title, training, testing) in enumerate([
            ("Original frozen MiniLM", original_train, original_test),
            (f"Tuned encoder: {encoder_name}", tuned_train, tuned_test)]):
        pca = PCA(2, random_state=42).fit(training)
        for row, (split, features, labels) in enumerate([
                ("Training (in-sample)", training, train_y), ("Test (unseen during fitting)", testing, y)]):
            points = pca.transform(features)
            axis = axes[row, column]
            for label in LABELS:
                take = labels == label
                axis.scatter(points[take, 0], points[take, 1], s=15, alpha=.7,
                             color=palette(int(label)), label=str(label))
            axis.set(title=f"{title}\n{split}", xlabel="PC1 (PCA fitted on that encoder's train embeddings)", ylabel="PC2")
    axes[0, 0].legend(title="True sentiment", fontsize=8)
    figure.suptitle("Genuine encoder embeddings; separate coordinate systems per encoder.\nLabels only color points; a 2D projection is not a test of class separability.", fontsize=11)
    figure.savefig(FINAL / "original_vs_tuned_embedding_pca.png", dpi=180)
    plt.close(figure)


def evaluate(device):
    lock, lock_digest = validate_lock()
    if (FINAL / "evaluation_complete.json").exists():
        done = json.loads((FINAL / "evaluation_complete.json").read_text())
        if done["selection_lock_sha256"] != lock_digest:
            raise ValueError("Completed test evaluation belongs to another selection")
        for name, expected in done["output_sha256"].items():
            if sha256(FINAL / name) != expected:
                raise ValueError(f"Completed test output changed: {name}")
        print(f"Prespecified evaluation already complete: {FINAL}; no test labels reread")
        return
    import torch
    import frozen_embeddings as frozen
    import encoder_finetuning as encoder
    from student_distillation import Student
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    selected = lock["selected"]
    final_encoder = RESULTS / "encoder/final_model"
    metadata = json.loads((final_encoder / "metadata.json").read_text())
    if metadata["config_name"] != selected["encoder"] or metadata["train_sha256"] != sha256(ROOT / "train.csv"):
        raise ValueError("Final fitted encoder does not match the immutable training selection")
    model_files = sorted(path for path in final_encoder.rglob("*") if path.is_file())
    model_hashes = {relative(path): sha256(path) for path in model_files}
    FINAL.mkdir(parents=True, exist_ok=True)
    started_path = FINAL / "evaluation_started.json"
    started = {"selection_lock_sha256": lock_digest, "final_encoder_sha256": model_hashes,
               "started_utc": datetime.now(timezone.utc).isoformat(), "device": device}
    if started_path.exists():
        previous = json.loads(started_path.read_text())
        if previous["selection_lock_sha256"] != lock_digest or previous["final_encoder_sha256"] != model_hashes:
            raise ValueError("An evaluation started with different locked artifacts; changing models after test exposure is forbidden")
    else:
        write_json(started_path, started, exclusive=True)
    # No labels are requested while producing any of the six test predictions.
    frame = pd.read_csv(ROOT / "test.csv", usecols=["PhraseId", "SentenceId", "Phrase"])
    train = pd.read_csv(ROOT / "train.csv")
    probabilities, features = {}, {}
    for name in dict.fromkeys(["minilm", selected["frozen"]]):
        features[name] = frozen.extract(name, frame, "test", 16, device)
    for output_name, name in [("minilm_frozen", "minilm"), ("selected_frozen", selected["frozen"])]:
        classifier = joblib.load(RESULTS / f"frozen/{name}/classifier.joblib")
        probabilities[output_name] = save_predictions(output_name, frame, classifier.predict_proba(features[name]))

    saved = torch.load(RESULTS / f"student/{selected['student']}/full_training_model.pt", map_location="cpu", weights_only=True)
    transformed = ((features["minilm"] - saved["scaler_mean"].numpy()) / saved["scaler_scale"].numpy()).astype(np.float32)
    student = Student(saved["n_features"])
    student.load_state_dict(saved["state_dict"])
    student.eval()
    with torch.no_grad():
        student_embeddings, logits = student(torch.from_numpy(transformed))
        probabilities["selected_student"] = save_predictions("selected_student", frame, logits.softmax(1).numpy())
    np.save(FINAL / "selected_student_test_embeddings.npy", student_embeddings.numpy())
    del student

    baseline = joblib.load(RESULTS / "baseline/selected_model.joblib")
    cleaned = pd.read_csv(ROOT / "Cleaned data/cleaned_test.csv", usecols=["PhraseId", "SentenceId", baseline["representation"]])
    if not np.array_equal(cleaned.PhraseId, frame.PhraseId) or not np.array_equal(cleaned.SentenceId, frame.SentenceId):
        raise ValueError("Cleaned test order does not match raw test order")
    probabilities["tfidf"] = save_predictions("tfidf", frame, baseline["pipeline"].predict_proba(cleaned[baseline["representation"]]))

    cfg = metadata["config"]
    model = encoder.SentimentEncoder(cfg["model"], cfg["last_layers"], path=final_encoder / "encoder")
    heads = torch.load(final_encoder / "heads.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(heads, strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError(f"Final encoder head mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(final_encoder / "encoder", local_files_only=True)
    encoded = tokenizer(frame.Phrase.astype(str).tolist(), truncation=True, padding=True,
                        max_length=metadata["max_length"], return_tensors="pt")
    probs, tuned_test, projected = encoder.predict(model, encoded, np.arange(len(frame)), torch.device(device))
    probabilities["selected_encoder"] = save_predictions("selected_encoder", frame, probs)
    np.save(FINAL / "selected_encoder_test_embeddings.npy", tuned_test)
    np.save(FINAL / "selected_encoder_test_projections.npy", projected)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    calibrated_frozen = scaled(probabilities["selected_frozen"], lock["temperature_scaling"]["selected_frozen"])
    calibrated_encoder = scaled(probabilities["selected_encoder"], lock["temperature_scaling"]["selected_encoder"])
    weight = lock["blend"]["encoder_weight"]
    probabilities["selected_blend"] = save_predictions("selected_blend", frame, weight * calibrated_encoder + (1 - weight) * calibrated_frozen)
    np.save(FINAL / "selected_frozen_calibrated_probabilities.npy", calibrated_frozen)
    np.save(FINAL / "selected_encoder_calibrated_probabilities.npy", calibrated_encoder)
    # All recipes and predictions above are fixed before the only scoring read.
    labels = pd.read_csv(ROOT / "test.csv", usecols=["PhraseId", "Sentiment"])
    if not np.array_equal(labels.PhraseId.to_numpy(), frame.PhraseId.to_numpy()):
        raise ValueError("Test-label row mismatch")
    y = labels.Sentiment.to_numpy(dtype=int)
    if set(y) != set(LABELS):
        raise ValueError("Unexpected test sentiment classes")
    scores = {name: probability_metrics(y, probabilities[name]) for name in COMPARATORS}
    calibration = {"temperature_fitting_source": "Training OOF only; temperatures locked before test",
                   "temperatures": lock["temperature_scaling"],
                   "selected_frozen_raw": scores["selected_frozen"],
                   "selected_frozen_calibrated": probability_metrics(y, calibrated_frozen),
                   "selected_encoder_raw": scores["selected_encoder"],
                   "selected_encoder_calibrated": probability_metrics(y, calibrated_encoder)}
    write_json(FINAL / "calibration.json", calibration)
    write_json(FINAL / "metrics.json", {"selection_lock_sha256": lock_digest, "primary": lock["primary"],
                                        "test_rows": len(y), "comparators": scores,
                                        "selection_uses_test_results": False})
    primary = lock["primary"]["procedure"]
    intervals = bootstrap(y, probabilities, primary)
    comparison = []
    for name in COMPARATORS:
        comparison.append({"procedure": name, "primary_chosen_from_training": name == primary,
                           **scalar_metrics(scores[name]),
                           "macro_f1_ci_lower": intervals["macro_f1"][name]["lower_95"],
                           "macro_f1_ci_upper": intervals["macro_f1"][name]["upper_95"]})
    pd.DataFrame(comparison).to_csv(FINAL / "comparison.csv", index=False)
    train_outputs = np.load(final_encoder / "train_outputs.npz")
    make_figures(y, scores, intervals, train.Sentiment.to_numpy(dtype=int),
                 np.load(ROOT / "experiments/features/minilm_train.npy"), features["minilm"],
                 train_outputs["embeddings"], tuned_test, selected["encoder"])
    output_files = sorted(path for path in FINAL.iterdir() if path.is_file() and path.name != "evaluation_complete.json")
    write_json(FINAL / "evaluation_complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_lock_sha256": lock_digest, "final_encoder_sha256": model_hashes,
        "test_sha256": sha256(ROOT / "test.csv"),
        "output_sha256": {path.name: sha256(path) for path in output_files},
        "selection_was_fixed_before_test": True}, exclusive=True)
    print(pd.DataFrame(comparison).to_string(index=False))
    print(json.dumps(intervals["primary_differences"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["select", "evaluate"])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    args = parser.parse_args()
    select() if args.command == "select" else evaluate(args.device)


if __name__ == "__main__":
    main()
