#!/usr/bin/env python3
"""Pool-based five-class sentiment active learning (simulation + human queue).

Uses previously extracted label-free features. No 341-review test.csv reads.
Simulation uses labels only as an oracle after acquisition; the stratified
cold-start is optimistic and is reported explicitly. Human queue accepts a
separate label CSV and never reads hidden labels from the pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("random", "entropy", "margin", "adjacent", "entropy_diversity", "hybrid", "balanced_hybrid")


def load_features(csv: Path, embedding: Path, rows_csv: Path | None):
    frame = pd.read_csv(csv)
    if not {"PhraseId", "Phrase"}.issubset(frame):
        raise ValueError("Pool must contain PhraseId and Phrase")
    if frame.PhraseId.isna().any() or frame.PhraseId.duplicated().any():
        raise ValueError("PhraseId must be unique and nonnull")
    features = np.load(embedding)
    if features.ndim != 2 or len(features) != len(frame) or not np.isfinite(features).all():
        raise ValueError("Embeddings must be finite and have one row per review")
    if rows_csv is None:
        raise ValueError("Provide --rows-csv to verify embedding/PhraseId alignment")
    rows = pd.read_csv(rows_csv)
    if "PhraseId" not in rows or not rows.PhraseId.astype(str).equals(frame.PhraseId.astype(str)):
        raise ValueError("Embedding row order does not match PhraseId order")
    return frame, normalize(features.astype(np.float32))


def fit_model(features, labels, indices, c=1.0):
    unique = set(labels[np.asarray(indices)].tolist())
    if unique != set(range(5)):
        raise ValueError("Need at least one verified labeled example of each class before fitting")
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=3000)
    model.fit(features[indices], labels[indices])
    return model


def query_batch(probs, embeddings, pool_indices, labeled_indices, batch_size, policy, rng, labeled_labels=None):
    """The query policy receives NO gold labels for the candidate pool."""
    pool_indices = np.asarray(pool_indices, dtype=int)
    labeled_indices = np.asarray(labeled_indices, dtype=int)
    n = min(batch_size, len(pool_indices))
    if n == 0:
        return np.array([], dtype=int)
    if policy == "random":
        return rng.choice(pool_indices, size=n, replace=False)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    if probs.shape != (len(pool_indices), 5):
        raise ValueError("Pool probabilities must have shape (pool_size, 5)")
    entropy = -(probs * np.log(probs)).sum(1) / np.log(5)
    adjacent = (np.sqrt(probs[:, :-1] * probs[:, 1:])
                * np.array([1.25, 1.0, 1.0, 1.25])).max(1)
    if policy in ("entropy", "entropy_diversity"):
        score = entropy
    elif policy in ("hybrid", "balanced_hybrid"):
        def rank(values):
            ordering = np.argsort(values, kind="stable")
            ranks = np.empty(len(values))
            ranks[ordering] = np.linspace(0., 1., len(values))
            return ranks
        base = .65 * rank(adjacent) + .35 * rank(entropy)
        if policy == "hybrid":
            score = base
        else:
            if labeled_labels is None or len(labeled_labels) != len(labeled_indices):
                raise ValueError("balanced_hybrid requires verified labels for labeled_indices only")
            known = np.asarray(labeled_labels, dtype=int)
            if not np.isin(known, range(5)).all():
                raise ValueError("Expected five-class verified labels")
            counts = np.bincount(known, minlength=5).astype(float)
            inverse = np.clip(np.sqrt(counts.mean() / (counts + 1.)), .7, 1.7)
            # Expected scarcity bonus; candidate pool truth is NEVER accessed.
            class_priority = (probs * inverse[None, :]).sum(1)
            score = .85 * base + .15 * rank(class_priority)
    elif policy == "margin":
        best_two = np.sort(probs, axis=1)[:, -2:]
        score = 1 - (best_two[:, 1] - best_two[:, 0])
    elif policy == "adjacent":
        score = adjacent
    else:
        raise ValueError(f"Unknown policy {policy!r}")
    if policy != "entropy_diversity":
        return pool_indices[np.argsort(-score, kind="stable")[:n]]
    # Greedy uncertainty + novelty; no y or evaluation labels are consulted.
    ranks = np.empty(len(score), dtype=float)
    ranks[np.argsort(score, kind="stable")] = np.linspace(0., 1., len(score))
    selected, used = [], np.zeros(len(score), dtype=bool)
    reference = embeddings[labeled_indices] if len(labeled_indices) else np.empty((0, embeddings.shape[1]))
    max_sim = ((embeddings[pool_indices] @ reference.T).max(1) if len(reference)
               else np.full(len(pool_indices), -1.))
    for _ in range(n):
        novelty = (1. - np.clip(max_sim, -1., 1.)) / 2.
        objective = .7 * ranks + .3 * novelty
        objective[used] = -np.inf
        pick = int(np.argmax(objective))
        used[pick] = True
        selected.append(pool_indices[pick])
        max_sim = np.maximum(max_sim, embeddings[pool_indices] @ embeddings[pool_indices[pick]])
    return np.asarray(selected, dtype=int)


def score(y, p):
    pred = p.argmax(1)
    rec = recall_score(y, pred, labels=list(range(5)), average=None, zero_division=0)
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)),
                **{f"recall_{i}": float(rec[i]) for i in range(5)})


def simulate(frame, features, policies, seeds, per_class, batch, rounds, c):
    if "Sentiment" not in frame or frame.Sentiment.isna().any():
        raise ValueError("Simulation requires gold Sentiment in train.csv")
    if frame.SentenceId.duplicated().any() if "SentenceId" in frame else False:
        raise ValueError("Duplicate SentenceId: use group-stratified split before simulation")
    y = frame.Sentiment.to_numpy(dtype=int)
    if set(y) != set(range(5)):
        raise ValueError("Expected five sentiment classes numbered 0..4")
    all_curves, all_queries = [], []
    for seed in seeds:
        rng_initial = np.random.default_rng(seed)
        available, holdout = train_test_split(np.arange(len(frame)), test_size=.2,
                                            stratify=y, random_state=seed)
        initial = np.sort(np.concatenate([rng_initial.choice(available[y[available] == cls],
                                             size=per_class, replace=False) for cls in range(5)]))
        for policy in policies:
            rng = np.random.default_rng(seed + 700)
            labeled = initial.copy()
            pending = np.setdiff1d(available, labeled)
            for round_index in range(rounds + 1):
                model = fit_model(features, y, labeled, c=c)
                p = model.predict_proba(features[holdout])
                all_curves.append(dict(seed=seed, policy=policy, round=round_index,
                                       labeled=len(labeled), **score(y[holdout], p)))
                if round_index == rounds or len(pending) == 0:
                    break
                probs = model.predict_proba(features[pending])
                chosen = query_batch(probs, features, pending, labeled, batch, policy, rng, y[labeled])
                for idx in chosen:
                    all_queries.append(dict(seed=seed, policy=policy,
                                            round=round_index + 1, row_index=int(idx),
                                            PhraseId=frame.iloc[idx].PhraseId,
                                            acquired_label=int(y[idx])))
                # Oracle reveal occurs AFTER policy selects indices.
                labeled = np.concatenate([labeled, chosen])
                pending = np.setdiff1d(pending, chosen)
    return pd.DataFrame(all_curves), pd.DataFrame(all_queries)


def human_queue(frame, features, annotations_csv, batch, policy, seed, c):
    if "Sentiment" in frame:
        raise ValueError("Human pool must NOT contain gold Sentiment labels")
    if annotations_csv.exists():
        annotated = pd.read_csv(annotations_csv)
        if not {"PhraseId", "Sentiment"}.issubset(annotated):
            raise ValueError("Annotations require PhraseId,Sentiment columns")
        if annotated.PhraseId.duplicated().any():
            raise ValueError("Duplicate annotations")
        annotated = annotated.dropna(subset=["Sentiment"])
        if not annotated.Sentiment.isin(range(5)).all():
            raise ValueError("Sentiment values must be integers 0..4")
        if not set(annotated.PhraseId.astype(str)).issubset(set(frame.PhraseId.astype(str))):
            raise ValueError("Annotation references a PhraseId absent from the pool")
        lookup = dict(zip(annotated.PhraseId.astype(str), annotated.Sentiment.astype(int)))
    else:
        lookup = {}
    labeled = np.array([i for i, id_ in enumerate(frame.PhraseId.astype(str)) if id_ in lookup], dtype=int)
    candidates = np.setdiff1d(np.arange(len(frame)), labeled)
    rng = np.random.default_rng(seed)
    if len(set(lookup.values())) < 5:
        if policy != "random" and len(labeled):
            print("Not all 5 classes have confirmed labels; selecting a random queue.")
        chosen = rng.choice(candidates, size=min(batch, len(candidates)), replace=False)
    else:
        y = np.array([lookup.get(id_, -1) for id_ in frame.PhraseId.astype(str)], dtype=int)
        model = fit_model(features, y, labeled, c)
        chosen = query_batch(model.predict_proba(features[candidates]), features, candidates,
                             labeled, batch, policy, rng, y[labeled])
    # No model guesses shown to annotators; avoid anchoring.
    queue = frame.iloc[chosen][["PhraseId", "Phrase"]].copy()
    queue["Sentiment"] = ""
    return queue, dict(labeled=len(labeled), queued=len(chosen), remaining=len(candidates))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("simulate", "human-next"))
    ap.add_argument("--csv", type=Path, default=ROOT / "train.csv")
    ap.add_argument("--embeddings", type=Path, default=ROOT / "experiments/features/bge_small_train.npy")
    ap.add_argument("--rows-csv", type=Path, default=ROOT / "experiments/features/bge_small_train_rows.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "experiments/results/active_learning_new")
    ap.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    ap.add_argument("--policy", choices=POLICIES, default="adjacent")
    ap.add_argument("--annotations", type=Path, default=Path("annotations.csv"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--c", type=float, default=1.0)
    args = ap.parse_args()
    frame, features = load_features(args.csv, args.embeddings, args.rows_csv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "simulate":
        curves, queries = simulate(frame, features, args.policies, args.seeds,
                                   args.per_class, args.batch, args.rounds, args.c)
        curves.to_csv(args.output / "curves.csv", index=False)
        queries.to_csv(args.output / "acquisitions.csv", index=False)
        summary = curves.groupby(["policy", "labeled"], as_index=False).agg(
            accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"))
        summary.to_csv(args.output / "summary.csv", index=False)
        print(summary[summary.labeled == summary.labeled.max()].to_string(index=False))
        # Paired final-budget differences + learning-curve area against random.
        final = curves[curves["round"] == curves["round"].max()]
        recall_cols = [f"recall_{i}" for i in range(5)]
        final.groupby("policy")[recall_cols].mean().to_csv(args.output / "class_recall.csv")
        baseline = final[final.policy == "random"].set_index("seed").macro_f1
        comparison = []
        for policy, part in final.groupby("policy"):
            differences = part.set_index("seed").macro_f1 - baseline
            mean = float(differences.mean())
            se = float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else float("nan")
            comparison.append(dict(policy=policy, seeds=len(differences), paired_gain=mean,
                                   normal_ci95_low=mean - 1.96 * se,
                                   normal_ci95_high=mean + 1.96 * se,
                                   wins_over_random=int((differences > 0).sum())))
        pd.DataFrame(comparison).to_csv(args.output / "paired_comparison.csv", index=False)
        # Per-seed normalized area under the macro-F1 learning curve.
        areas = []
        for (seed, policy), part in curves.groupby(["seed", "policy"]):
            part = part.sort_values("labeled")
            area = float(np.trapz(part.macro_f1, part.labeled)
                         / (part.labeled.iloc[-1] - part.labeled.iloc[0]))
            areas.append(dict(seed=seed, policy=policy, normalized_auc_macro_f1=area))
        pd.DataFrame(areas).to_csv(args.output / "learning_curve_auc.csv", index=False)
    else:
        queue, status = human_queue(frame, features, args.annotations,
                                    args.batch, args.policy, args.seeds[0], args.c)
        queue.to_csv(args.output / "annotation_batch.csv", index=False)
        (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(status)


if __name__ == "__main__":
    main()
