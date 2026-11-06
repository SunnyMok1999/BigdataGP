"""Five-class sentiment classification with a measured MaleCNS sparse recurrent mask.

Train-only benchmark. Does not load test.csv or choose epochs on outer validation.
The graph is a documented 80-neuron MaleCNS v1.0 induced subcircuit.
See experiments/MaleCNS_README.md for data provenance and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GRAPH = HERE / "malecns_data" / "graph.json"
LABELS = np.arange(5)
VOCAB = 8193
TOKENS = re.compile(r"[a-z0-9]+|[!?]", re.I)


def tokenize(text: str, max_length: int = 48) -> list[int]:
    """Stable unsupervised hashing: 0 is padding; no fold/test vocabulary fitting."""
    tokens = TOKENS.findall(str(text).lower())[:max_length]
    return [
        1 + int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little") % (VOCAB - 1)
        for w in tokens
    ] or [1]


def tensorize(texts: list[str], max_length: int):
    encoded = [tokenize(t, max_length) for t in texts]
    lengths = np.asarray([len(e) for e in encoded], dtype=np.int64)
    padded = np.zeros((len(encoded), int(lengths.max())), dtype=np.int64)
    for i, e in enumerate(encoded):
        padded[i, :len(e)] = e
    return torch.tensor(padded), torch.tensor(lengths)


def read_measured_graph(path: Path, n_nodes: int):
    """Select high-contact nodes without using review labels; retain measured edges."""
    data = json.loads(path.read_text())
    nodes = data["nodes"]
    edges = data["edges"]
    if not 4 <= n_nodes <= len(nodes):
        raise ValueError(f"nodes must be between 4 and {len(nodes)}")
    if any(len(e) < 3 for e in edges):
        raise ValueError("Expected source-index, target-index, contact-count triples")
    degree = np.zeros(len(nodes), dtype=np.int64)
    for source, target, contacts in edges:
        degree[int(source)] += int(contacts)
        degree[int(target)] += int(contacts)
    selected = sorted(
        sorted(range(len(nodes)), key=lambda i: (-degree[i], int(nodes[i]["id"])))[:n_nodes]
    )
    inverse = {old: new for new, old in enumerate(selected)}
    pairs = {(inverse[int(s)], inverse[int(t)]) for s, t, w in edges
             if int(s) in inverse and int(t) in inverse and int(w) > 0}
    if len(pairs) < n_nodes:
        raise ValueError("Insufficient induced edges in selected subgraph")
    manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return pairs, selected, manifest_hash


def directed_rewire(edges: set[tuple[int, int]], seed: int, swaps_per_edge: int = 10):
    """Directed double-edge switches preserve every node's in/out degree."""
    rng = random.Random(seed)
    result = set(edges)
    seq = sorted(result)
    if len(seq) < 2:
        return result
    for _ in range(swaps_per_edge * len(seq)):
        i, j = rng.sample(range(len(seq)), 2)
        a, b = seq[i]
        c, d = seq[j]
        ab, cd = (a, d), (c, b)
        if a == c or b == d or a == d or c == b:
            continue
        if ab in result or cd in result:
            continue
        result.remove((a, b))
        result.remove((c, d))
        result.update((ab, cd))
        seq[i], seq[j] = ab, cd
    return result


def random_edges(n: int, m: int, seed: int):
    """Same nodes/edges but not degree preserving; no new self loops."""
    rng = np.random.default_rng(seed)
    eligible = [(a, b) for a in range(n) for b in range(n) if a != b]
    if m > len(eligible):
        raise ValueError("Too many edges for a self-loop-free graph")
    return {eligible[i] for i in rng.choice(len(eligible), size=m, replace=False)}


def mask_for(measured: set[tuple[int, int]], n: int, kind: str, seed: int):
    if kind == "biological":
        pairs = measured
    elif kind == "degree_rewired":
        pairs = directed_rewire(measured, seed)
    elif kind == "edge_random":
        pairs = random_edges(n, len(measured), seed)
    else:
        raise ValueError(f"Unknown graph kind: {kind}")
    # Row = postsynaptic target, column = presynaptic source.
    mask = np.zeros((n, n), np.float32)
    for source, target in pairs:
        mask[target, source] = 1.
    if int(mask.sum()) != len(measured):
        raise AssertionError("Graph edge count changed")
    return mask, pairs


class SparseConnectomeRNN(nn.Module):
    """Only the measured graph's recurrent nonzero entries are trainable.

    Word embeddings, input projection and classifier are learned as in ordinary
    NLP models. Topology is not a claim about physiological brain computation.
    """
    def __init__(self, mask: np.ndarray, embed_dim: int = 48, dropout: float = .25):
        super().__init__()
        n = int(mask.shape[0])
        self.register_buffer("mask", torch.tensor(mask.copy()))
        self.embedding = nn.Embedding(VOCAB, embed_dim, padding_idx=0)
        self.input_layer = nn.Linear(embed_dim, n)
        self.recurrent = nn.Parameter(torch.empty(n, n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(n * 2), nn.Linear(n * 2, 5))
        nn.init.normal_(self.recurrent, std=0.08)
        with torch.no_grad():
            self.recurrent.mul_(self.mask)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, steps = tokens.shape
        x = self.dropout(self.embedding(tokens))
        h = x.new_zeros(batch, self.mask.shape[0])
        total = torch.zeros_like(h)
        w = self.recurrent * self.mask
        for t in range(steps):
            new_h = torch.tanh(self.input_layer(x[:, t]) + F.linear(h, w, self.bias))
            active = (t < lengths).unsqueeze(1)
            h = torch.where(active, new_h, h)
            total = total + h * active
        pooled = total / lengths.clamp_min(1).unsqueeze(1).to(x.dtype)
        return self.head(self.dropout(torch.cat([h, pooled], dim=1)))


def one_fold(tokens, lengths, y, train_ids, val_ids, mask, *, seed, epochs,
             batch_size, lr, device, dropout):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SparseConnectomeRNN(mask, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_ids = np.asarray(train_ids)
    val_ids = np.asarray(val_ids)
    # All class weights use only the outer training fold.
    counts = np.bincount(y[train_ids], minlength=5).astype(np.float32)
    weights = np.sqrt(max(1., len(train_ids)) / (5. * counts.clip(min=1)))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    gen = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for batch in np.array_split(gen.permutation(train_ids), max(1, int(np.ceil(len(train_ids) / batch_size)))):
            xb, lb = tokens[batch].to(device), lengths[batch].to(device)
            target = torch.tensor(y[batch], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb, lb), target, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in np.array_split(val_ids, max(1, int(np.ceil(len(val_ids) / batch_size)))):
            probs = model(tokens[batch].to(device), lengths[batch].to(device)).softmax(1)
            outputs.append(probs.cpu().numpy())
    return np.concatenate(outputs)


def evaluate(args):
    if args.epochs < 1 or args.folds < 2:
        raise ValueError("epochs >= 1 and folds >= 2 required")
    if args.device == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    data = pd.read_csv(args.train)
    required = {"Phrase", "Sentiment"}
    if not required.issubset(data.columns):
        raise ValueError(f"Training CSV must contain {required}")
    if data["Phrase"].isna().any() or data["Sentiment"].isna().any():
        raise ValueError("Training text and labels cannot be null")
    y = data.Sentiment.to_numpy(dtype=int)
    if set(y) != set(LABELS):
        raise ValueError("Expected exactly the five sentiment classes 0..4")
    if np.bincount(y).min() < args.folds:
        raise ValueError("Each class must contain at least fold_count examples")
    tokens, lengths = tensorize(data.Phrase.astype(str).tolist(), args.max_length)
    selected, original_ids, sha = read_measured_graph(Path(args.graph_path), args.nodes)
    kinds = args.graphs.split(",")
    if len(set(kinds)) != len(kinds):
        raise ValueError("Duplicate graph kinds")
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(tokens, y))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"data_path": str(args.train), "training_rows": len(y),
            "protocol": "training-only stratified OOF; fixed epochs; no test/validation tuning",
            "measured_graph_file_sha256": sha, "original_neuron_indices": original_ids,
            "n_nodes": args.nodes, "measured_edges": len(selected), "seeds": args.seeds,
            "graph_types": kinds, "epochs": args.epochs, "folds": args.folds}
    (out / "run_config.json").write_text(json.dumps(meta, indent=2))
    metrics, predictions = [], []
    for seed in args.seeds:
        for kind in kinds:
            oof = np.zeros((len(y), 5), np.float32)
            graph_mask, graph_edges = mask_for(selected, args.nodes, kind, seed)
            for fold, (train_ids, val_ids) in enumerate(folds):
                probs = one_fold(tokens, lengths, y, train_ids, val_ids, graph_mask,
                                 seed=seed + fold * 101, epochs=args.epochs,
                                 batch_size=args.batch_size, lr=args.lr,
                                 device=args.device, dropout=args.dropout)
                oof[val_ids] = probs
                pred = probs.argmax(1)
                metrics.append({"seed": seed, "graph": kind, "fold": fold,
                                "nodes": args.nodes, "edges": len(graph_edges),
                                "accuracy": accuracy_score(y[val_ids], pred),
                                "macro_f1": f1_score(y[val_ids], pred, average="macro", zero_division=0)})
            pred = oof.argmax(1)
            for i in range(len(y)):
                predictions.append({"seed": seed, "graph": kind,
                                    "row_index": i, "prediction": int(pred[i]),
                                    **{f"prob_{j}": float(oof[i, j]) for j in LABELS}})
            print(f"seed={seed} {kind}: acc={accuracy_score(y,pred):.4f} "
                  f"macroF1={f1_score(y,pred,average='macro',zero_division=0):.4f}", flush=True)
            recall = recall_score(y, pred, labels=LABELS, average=None, zero_division=0)
            meta[f"seed_{seed}_{kind}"] = {
                "accuracy_oof": accuracy_score(y, pred),
                "macro_f1_oof": f1_score(y, pred, average="macro", zero_division=0),
                "class_recall": recall.tolist(),
                "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
            }
    pd.DataFrame(metrics).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "oof_predictions.csv", index=False)
    means = pd.DataFrame(metrics).groupby("graph")[["accuracy","macro_f1"]].agg(["mean","std"])
    means.to_csv(out / "summary.csv")
    (out / "run_config.json").write_text(json.dumps(meta, indent=2))
    print(means.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=str(ROOT / "train.csv"))
    p.add_argument("--graph-path", default=str(GRAPH))
    p.add_argument("--nodes", type=int, default=80, choices=(32, 64, 80))
    p.add_argument("--graphs", default="biological,degree_rewired,edge_random")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=48)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--device", choices=("cpu","cuda","mps"), default="cpu")
    p.add_argument("--output", default=str(HERE / "results" / "malecns"))
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
