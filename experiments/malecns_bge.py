"""Train-only comparison: frozen BGE sentence vectors -> MaleCNS recurrent reservoir.

Unlike the token-sequence classifier, this iterates a directed recurrent
network under a CONSTANT BGE sentence input; iteration steps are not words.
This allows direct comparison with the repo's existing BGE-small+LR baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F

from malecns_sentiment import GRAPH, LABELS, ROOT, mask_for, read_measured_graph


class BGESparseRecurrent(nn.Module):
    def __init__(self, mask: np.ndarray, input_dim: int, steps: int = 6):
        super().__init__()
        n = mask.shape[0]
        self.register_buffer("mask", torch.tensor(mask.copy()))
        self.drive = nn.Linear(input_dim, n)
        self.weights = nn.Parameter(torch.empty(n, n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.steps = steps
        self.dropout = nn.Dropout(.25)
        self.head = nn.Sequential(nn.LayerNorm(2 * n), nn.Linear(2 * n, 5))
        nn.init.normal_(self.weights, std=.08)
        with torch.no_grad():
            self.weights.mul_(self.mask)

    def forward(self, x: torch.Tensor):
        z = self.drive(self.dropout(x))
        h = z.new_zeros(z.shape)
        total = torch.zeros_like(h)
        W = self.weights * self.mask
        for _ in range(self.steps):
            h = .5 * h + .5 * torch.tanh(z + F.linear(h, W, self.bias))
            total = total + h
        pooled = total / self.steps
        return self.head(self.dropout(torch.cat((h, pooled), dim=1)))


def load_bge(train: Path, feature_path: Path, rows_path: Path):
    df = pd.read_csv(train)
    rows = pd.read_csv(rows_path)
    if not all(c in df for c in ("PhraseId", "Sentiment", "Phrase")):
        raise ValueError("train.csv must contain PhraseId, Sentiment, Phrase")
    if len(df) != len(rows) or not np.array_equal(df.PhraseId, rows.PhraseId):
        raise ValueError("BGE features and train.csv PhraseId order differ")
    if "Sentiment" in rows and not np.array_equal(df.Sentiment, rows.Sentiment):
        raise ValueError("BGE row sentiment labels do not match train.csv")
    if not df.PhraseId.is_unique:
        raise ValueError("Expected unique review PhraseId")
    X = np.load(feature_path, allow_pickle=False)
    if X.shape != (len(df), 384) or not np.isfinite(X).all():
        raise ValueError(f"Expected finite BGE-small features (N,384), got {X.shape}")
    y = df.Sentiment.to_numpy(dtype=int)
    if set(y) != set(LABELS):
        raise ValueError("Expected labels 0..4")
    return X.astype(np.float32), y


def train_predict(X, y, train_ids, val_ids, mask, *, seed, epochs, steps, lr, batch_size, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BGESparseRecurrent(mask, input_dim=X.shape[1], steps=steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.05)
    Xtrain = torch.tensor(X[train_ids], device=device)
    ytrain = torch.tensor(y[train_ids], dtype=torch.long, device=device)
    counts = np.bincount(y[train_ids], minlength=5).astype(np.float32)
    balancing = torch.tensor(np.sqrt(len(train_ids) / (5 * counts.clip(min=1))),
                             device=device)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        idx = rng.permutation(len(train_ids))
        for batch in np.array_split(idx, max(1, int(np.ceil(len(idx) / batch_size)))):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(Xtrain[batch]), ytrain[batch], weight=balancing)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            model(torch.tensor(X[batch], device=device)).softmax(1).cpu().numpy()
            for batch in np.array_split(val_ids, max(1, int(np.ceil(len(val_ids) / batch_size))))
        ])


def run(args):
    if args.device == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    X, y = load_bge(Path(args.train), Path(args.features), Path(args.rows))
    graph, ids, sha = read_measured_graph(Path(args.graph_path), args.nodes)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics = []
    summary = {"source_graph_sha256": sha, "nodes": args.nodes,
               "neuron_source_indices": ids, "edges": len(graph),
               "seed_list": args.seeds, "epochs": args.epochs, "steps": args.steps,
               "method": "5-fold fixed-epoch OOF using frozen BGE-small 384D embeddings, no test labels"}
    predictions = []
    for seed in args.seeds:
        for kind in ("biological", "degree_rewired", "edge_random"):
            mask, edges = mask_for(graph, args.nodes, kind, seed)
            oof = np.zeros((len(y), 5), np.float32)
            for fold, (tr, va) in enumerate(folds):
                oof[va] = train_predict(X, y, tr, va, mask, seed=seed + fold * 101,
                                        epochs=args.epochs, steps=args.steps,
                                        lr=args.lr, batch_size=args.batch_size, device=args.device)
                pred = oof[va].argmax(1)
                metrics.append({"seed": seed, "graph": kind, "fold": fold,
                                "accuracy": accuracy_score(y[va], pred),
                                "macro_f1": f1_score(y[va], pred, average="macro", zero_division=0)})
            pred = oof.argmax(1)
            key = f"seed_{seed}_{kind}"
            summary[key] = {
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "class_recall": recall_score(y, pred, labels=LABELS, average=None, zero_division=0).tolist(),
                "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist()
            }
            for i in range(len(y)):
                predictions.append({"seed": seed, "graph": kind, "row_index": i, "prediction": int(pred[i]),
                                    **{f"prob_{c}": float(oof[i, c]) for c in LABELS}})
            print(f"BGE seed={seed} graph={kind} acc={summary[key]['accuracy']:.4f} "
                  f"macroF1={summary[key]['macro_f1']:.4f}", flush=True)
    pd.DataFrame(metrics).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(metrics).groupby("graph")[["accuracy", "macro_f1"]].agg(["mean", "std"]).to_csv(out / "summary.csv")
    (out / "run_config.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "train.csv"))
    parser.add_argument("--features", default=str(ROOT / "experiments/features/bge_small_train.npy"))
    parser.add_argument("--rows", default=str(ROOT / "experiments/features/bge_small_train_rows.csv"))
    parser.add_argument("--graph-path", default=str(GRAPH))
    parser.add_argument("--nodes", choices=(32, 64, 80), type=int, default=80)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--device", default="cpu", choices=("cpu","mps","cuda"))
    parser.add_argument("--output", default=str(ROOT / "experiments/results/malecns_bge"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
