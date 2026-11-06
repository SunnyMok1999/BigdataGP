"""Controlled five-class movie-sentiment comparison of MaleCNS, vanilla RNN and LSTM.

This is a train.csv-only, fixed-epoch cross-validation experiment.
Uses the same hash tokenizer, embedding size, folds, class weighting and
training schedule as experiments/malecns_sentiment.py.
The full-width RNN/LSTM recurrent matrices have different parameter counts
from the sparse MaleCNS mask; the experiment is an architecture comparison,
not a parameter-matched causal estimate of topology.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from malecns_sentiment import (
    GRAPH, LABELS, ROOT, VOCAB, SparseConnectomeRNN,
    mask_for, read_measured_graph, tensorize,
)

HERE = Path(__file__).resolve().parent


class StandardSequenceClassifier(nn.Module):
    """Dense, one-layer, unidirectional tanh RNN or LSTM over word tokens."""

    def __init__(self, kind: str, hidden_size: int = 80,
                 embed_dim: int = 48, dropout: float = 0.25):
        super().__init__()
        if kind not in {"rnn", "lstm"}:
            raise ValueError("kind must be 'rnn' or 'lstm'")
        self.kind = kind
        self.embedding = nn.Embedding(VOCAB, embed_dim, padding_idx=0)
        cell = nn.RNN if kind == "rnn" else nn.LSTM
        self.recurrent = cell(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            **({"nonlinearity": "tanh"} if kind == "rnn" else {}),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 5),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or lengths.ndim != 1 or len(lengths) != len(tokens):
            raise ValueError("Expected tokens [B,T] and lengths [B]")
        x = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, final_state = self.recurrent(packed)
        padded, _ = pad_packed_sequence(
            outputs, batch_first=True, total_length=tokens.shape[1]
        )
        final_hidden = final_state[0][-1] if self.kind == "lstm" else final_state[-1]
        time_mask = (torch.arange(tokens.shape[1], device=tokens.device)[None, :]
                     < lengths[:, None])
        pooled = (padded * time_mask.unsqueeze(-1)).sum(1) / lengths.clamp_min(1).unsqueeze(1)
        return self.head(self.dropout(torch.cat([final_hidden, pooled], dim=1)))


def run_fold(model_kind, graph_mask, tokens, lengths, labels, train_ids, val_ids,
             *, seed, epochs, batch_size, lr, device, dropout, hidden_size):
    """Fixed epoch count; never use held-out fold labels to select the model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model_kind == "malecns":
        if graph_mask is None:
            raise ValueError("Measured graph mask required for MaleCNS")
        model = SparseConnectomeRNN(graph_mask, embed_dim=48, dropout=dropout).to(device)
    else:
        model = StandardSequenceClassifier(
            model_kind, hidden_size=hidden_size, embed_dim=48, dropout=dropout
        ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_ids = np.asarray(train_ids, dtype=int)
    val_ids = np.asarray(val_ids, dtype=int)
    counts = np.bincount(labels[train_ids], minlength=5).astype(np.float32)
    weights = torch.tensor(
        np.sqrt(len(train_ids) / (5.0 * counts.clip(min=1))), device=device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    generator = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = generator.permutation(train_ids)
        chunks = np.array_split(order, int(np.ceil(len(order) / batch_size)))
        for chunk in chunks:
            xb = tokens[chunk].to(device)
            lb = lengths[chunk].to(device)
            yb = torch.as_tensor(labels[chunk], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, lb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            opt.step()
    model.eval()
    probabilities = []
    with torch.no_grad():
        chunks = np.array_split(val_ids, int(np.ceil(len(val_ids) / batch_size)))
        for chunk in chunks:
            p = model(tokens[chunk].to(device), lengths[chunk].to(device)).softmax(1)
            probabilities.append(p.cpu().numpy())
    return np.concatenate(probabilities), parameter_count


def evaluate(args):
    if args.epochs < 1 or args.folds < 2 or args.hidden_size < 4:
        raise ValueError("epochs >=1, folds >=2 and hidden-size >=4 required")
    if len(args.seeds) != len(set(args.seeds)) or len(args.models) != len(set(args.models)):
        raise ValueError("Duplicate seeds or model kinds")
    if args.device == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    train = pd.read_csv(args.train)
    required = {"PhraseId", "Phrase", "Sentiment"}
    if not required.issubset(train.columns):
        raise ValueError(f"Training CSV requires {required}")
    if train[list(required)].isna().any().any() or not train.PhraseId.is_unique:
        raise ValueError("Null training rows or duplicate PhraseId")
    y = train.Sentiment.to_numpy(dtype=int)
    if set(y) != set(LABELS) or np.bincount(y).min() < args.folds:
        raise ValueError("Five classes with enough examples for each fold required")

    tokens, lengths = tensorize(train.Phrase.astype(str).tolist(), args.max_length)
    mask = None
    graph_sha = None
    if "malecns" in args.models:
        measured, _, graph_sha = read_measured_graph(Path(args.graph), args.hidden_size)
        mask, _ = mask_for(measured, args.hidden_size, "biological", 42)
    folds = list(StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=42
    ).split(np.arange(len(y)), y))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fold_metrics = []
    per_seed_metrics = []
    oof_rows = []
    for seed in args.seeds:
        for kind in args.models:
            probabilities = np.zeros((len(y), 5), dtype=np.float32)
            fold_assignment = np.full(len(y), -1, dtype=int)
            count = None
            for fold, (train_idx, val_idx) in enumerate(folds):
                p, count = run_fold(
                    kind, mask, tokens, lengths, y, train_idx, val_idx,
                    seed=seed + 101 * fold,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, device=args.device, dropout=args.dropout,
                    hidden_size=args.hidden_size,
                )
                probabilities[val_idx] = p
                fold_assignment[val_idx] = fold
                pred_fold = p.argmax(1)
                fold_metrics.append({
                    "model": kind, "seed": seed, "fold": fold, "size": len(val_idx),
                    "accuracy": accuracy_score(y[val_idx], pred_fold),
                    "macro_f1": f1_score(y[val_idx], pred_fold, average="macro", zero_division=0),
                    "parameters": count,
                })
            pred = probabilities.argmax(1)
            score = {
                "model": kind, "seed": seed, "parameters": count,
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            }
            for cls, recall in zip(
                LABELS, recall_score(y, pred, labels=LABELS, average=None, zero_division=0)
            ):
                score[f"recall_{cls}"] = float(recall)
            score["confusion_matrix"] = confusion_matrix(y, pred, labels=LABELS).tolist()
            per_seed_metrics.append(score)
            for i in range(len(y)):
                oof_rows.append({
                    "model": kind, "seed": seed, "row_index": i,
                    "PhraseId": int(train.PhraseId.iloc[i]),
                    "outer_fold": int(fold_assignment[i]),
                    "prediction": int(pred[i]),
                    **{f"prob_{cls}": float(probabilities[i, cls]) for cls in LABELS},
                })
            print(f"model={kind} seed={seed} accuracy={score['accuracy']:.4f} "
                  f"macroF1={score['macro_f1']:.4f} parameters={count}", flush=True)

    pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "oof_predictions.csv", index=False)
    summary_frame = pd.DataFrame([{k: v for k, v in m.items() if k != "confusion_matrix"}
                                  for m in per_seed_metrics])
    summary_frame.to_csv(out / "per_seed_metrics.csv", index=False)
    summary = summary_frame.groupby("model").agg(
        mean_macro_f1=("macro_f1", "mean"), std_macro_f1=("macro_f1", "std"),
        mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        parameters=("parameters", "first"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "method": "stratified, training-only fixed-epoch OOF; no test access",
        "train_path": str(args.train), "n_train": len(y),
        "models": args.models, "seeds": args.seeds, "folds": args.folds,
        "epochs": args.epochs, "embedding_dim": 48,
        "hidden_size": args.hidden_size, "max_length": args.max_length,
        "learning_rate": args.lr, "batch_size": args.batch_size,
        "dropout": args.dropout, "male_cns_graph_sha256": graph_sha,
        "per_seed": per_seed_metrics,
        "limitation": "Dense RNN/LSTM parameter counts differ from sparse MaleCNS; exploratory architecture controls.",
    }, indent=2))
    print(summary.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=str(ROOT / "train.csv"))
    p.add_argument("--graph", default=str(GRAPH))
    p.add_argument("--models", nargs="+", default=["malecns", "rnn", "lstm"],
                   choices=["malecns", "rnn", "lstm"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--hidden-size", type=int, default=80)
    p.add_argument("--max-length", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--output", default=str(HERE / "results" / "rnn_lstm"))
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
