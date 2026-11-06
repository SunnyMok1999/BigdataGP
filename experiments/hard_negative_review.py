#!/usr/bin/env python3
"""Create human review queues from TRAIN-only out-of-fold errors.

Never train on candidates until their labels/semantics have been verified.
Pairs generated globally are for AUDIT only; in CV their sample IDs must be
filtered to the current training fold, with hard mining rerun inside that fold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
LABELS = list(range(5))


def verified_alignment(frame, predictions, rows):
    required = {"PhraseId", "Sentiment", "prediction", *{f"prob_{j}" for j in LABELS}}
    if not required.issubset(predictions):
        raise ValueError(f"OOF predictions missing {required - set(predictions.columns)}")
    for table, name in ((predictions, "OOF predictions"), (rows, "embedding rows")):
        if "PhraseId" not in table or not np.array_equal(
            frame.PhraseId.to_numpy().astype(str), table.PhraseId.to_numpy().astype(str)):
            raise ValueError(f"{name} PhraseId order does not match train.csv")
    if not np.array_equal(frame.Sentiment.to_numpy(), predictions.Sentiment.to_numpy()):
        raise ValueError("OOF labels do not match train.csv")
    if "outer_fold" not in predictions or predictions.outer_fold.isna().any():
        raise ValueError("Expected outer-fold predictions, not in-sample predictions")
    probabilities = predictions[[f"prob_{j}" for j in LABELS]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1., atol=1e-4):
        raise ValueError("Invalid five-class probabilities")
    return np.clip(probabilities, 1e-10, 1.0)


def make_queues(frame, pred, vectors, per_boundary=25, aug_count=60):
    y = frame.Sentiment.to_numpy(int)
    p = np.clip(pred[[f"prob_{j}" for j in LABELS]].to_numpy(float), 1e-10, 1.)
    guesses = p.argmax(1)
    loss = -np.log(p[np.arange(len(y)), y])
    entropy = -(p * np.log(p)).sum(1) / np.log(5.)
    ranks = .7 * pd.Series(loss).rank(pct=True).to_numpy() + .3 * pd.Series(entropy).rank(pct=True).to_numpy()
    audit = frame[["PhraseId", "SentenceId", "Phrase", "Sentiment"]].copy()
    audit["predicted"] = guesses
    audit["confidence"] = p.max(1)
    audit["gold_probability"] = p[np.arange(len(y)), y]
    audit["cross_entropy"] = loss
    audit["entropy"] = entropy
    audit["difficulty_rank"] = ranks
    audit["adjacent_mistake"] = (np.abs(guesses - y) == 1)
    audit["suspected_noise"] = (guesses != y) & (p.max(1) >= .55)
    audit["reviewed_label"] = ""
    audit["label_verified"] = False
    audit["notes"] = ""
    audit = audit.sort_values("difficulty_rank", ascending=False).reset_index(drop=True)

    vectors = normalize(vectors.astype(np.float32))
    similarity = vectors @ vectors.T
    triples = []
    for boundary in range(4):
        candidates = np.where((np.isin(y, [boundary, boundary + 1])) &
                              (np.abs(guesses - y) == 1))[0]
        candidates = sorted(candidates, key=lambda idx: -ranks[idx])[:per_boundary]
        for i in candidates:
            positives = np.where(y == y[i])[0]
            positives = positives[positives != i]
            negatives = np.where(y == (boundary + 1 if y[i] == boundary else boundary))[0]
            if not len(positives) or not len(negatives):
                continue
            pos = int(positives[similarity[i, positives].argmax()])
            neg = int(negatives[similarity[i, negatives].argmax()])
            triples.append({
                "boundary": f"{boundary}-{boundary+1}",
                "anchor_id": frame.iloc[i].PhraseId, "anchor_label": int(y[i]),
                "anchor_text": frame.iloc[i].Phrase,
                "positive_id": frame.iloc[pos].PhraseId, "positive_label": int(y[pos]),
                "positive_text": frame.iloc[pos].Phrase,
                "negative_id": frame.iloc[neg].PhraseId, "negative_label": int(y[neg]),
                "negative_text": frame.iloc[neg].Phrase,
                "negative_cosine": float(similarity[i, neg]),
                "anchor_oof_prediction": int(guesses[i]),
                "approved": False, "labels_verified": False, "intensity_difference_verified": False,
                "review_notes": "",
            })
    triplets = pd.DataFrame(triples)
    difficult = audit[(audit.adjacent_mistake) & (~audit.suspected_noise)].head(aug_count)
    prompts = difficult[["PhraseId", "Phrase", "Sentiment", "difficulty_rank"]].copy()
    prompts["instruction"] = (
        "Paraphrase the review without changing polarity, intensity, negation, "
        "contrast, or the 0..4 sentiment label. Do not invent new facts. "
        "A human must check the result before training.")
    prompts["augmented_text"] = ""
    prompts["approved"] = False
    prompts["label_verified"] = False
    prompts["notes"] = ""
    return audit, triplets, prompts


def export_approved(aug_file: Path, triplet_file: Path, output: Path, train: Path):
    frame = pd.read_csv(train)
    originals = dict(zip(frame.PhraseId.astype(str), frame.Sentiment.astype(int)))
    aug = pd.read_csv(aug_file).fillna("")
    pairs = pd.read_csv(triplet_file).fillna("")
    required_aug = {"PhraseId", "Sentiment", "augmented_text", "approved", "label_verified"}
    required_pairs = {"anchor_id", "positive_id", "negative_id", "anchor_label",
                      "positive_label", "negative_label", "approved", "labels_verified",
                      "intensity_difference_verified"}
    if not required_aug.issubset(aug) or not required_pairs.issubset(pairs):
        raise ValueError("Review files are missing required columns")
    def yes(col):
        return col.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    aug = aug[yes(aug.approved) & yes(aug.label_verified)].copy()
    pairs = pairs[yes(pairs.approved) & yes(pairs.labels_verified) &
                  yes(pairs.intensity_difference_verified)].copy()
    for row in aug.itertuples():
        key = str(row.PhraseId)
        if key not in originals or originals[key] != int(row.Sentiment):
            raise ValueError(f"Approved augmentation label mismatch: {key}")
        if len(str(row.augmented_text).strip()) < 8 or str(row.augmented_text).strip() == str(row.Phrase).strip():
            raise ValueError(f"Augmentation lacks a distinct usable paraphrase: {key}")
    for row in pairs.itertuples():
        keys = [str(row.anchor_id), str(row.positive_id), str(row.negative_id)]
        if len(set(keys)) != 3 or not all(key in originals for key in keys):
            raise ValueError(f"Invalid reviewed triplet ids: {keys}")
        labels = [originals[key] for key in keys]
        if labels != [int(row.anchor_label), int(row.positive_label), int(row.negative_label)]:
            raise ValueError(f"Triplet label mismatch: {keys}")
        if labels[0] != labels[1] or abs(labels[0] - labels[2]) != 1:
            raise ValueError(f"Triplet must contrast adjacent classes: {keys}")
    output.mkdir(parents=True, exist_ok=True)
    aug.to_csv(output / "augmented_APPROVED.csv", index=False)
    pairs.to_csv(output / "triplets_APPROVED.csv", index=False)
    return len(aug), len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["mine", "export-approved"])
    parser.add_argument("--train", type=Path, default=ROOT / "train.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "experiments/features/bge_small_train.npy")
    parser.add_argument("--rows-csv", type=Path, default=ROOT / "experiments/features/bge_small_train_rows.csv")
    parser.add_argument("--oof", type=Path, default=ROOT / "experiments/results/frozen/bge_small/nested_oof_predictions.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/hard_negative_review")
    parser.add_argument("--per-boundary", type=int, default=25)
    parser.add_argument("--aug-count", type=int, default=60)
    parser.add_argument("--aug-file", type=Path)
    parser.add_argument("--triplet-file", type=Path)
    args = parser.parse_args()
    if args.command == "export-approved":
        if not args.aug_file or not args.triplet_file:
            parser.error("export-approved requires --aug-file and --triplet-file")
        print("Exported approved augmentation/pairs:", export_approved(
            args.aug_file, args.triplet_file, args.out, args.train))
        return
    frame, preds, rows = pd.read_csv(args.train), pd.read_csv(args.oof), pd.read_csv(args.rows_csv)
    verified_alignment(frame, preds, rows)
    features = np.load(args.features)
    if features.shape[0] != len(frame) or not np.isfinite(features).all():
        raise ValueError("Embedding shape or data invalid")
    audit, pairs, aug = make_queues(frame, preds, features, args.per_boundary, args.aug_count)
    args.out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out / "label_audit_REVIEW.csv", index=False)
    pairs.to_csv(args.out / "adjacent_triplets_REVIEW.csv", index=False)
    aug.to_csv(args.out / "augmentation_REVIEW.csv", index=False)
    print(f"Created review queues: {len(audit)} audit rows, {len(pairs)} pairs, {len(aug)} paraphrase requests")


if __name__ == "__main__":
    main()
