"""Tests for existing frozen BGE cache and a connectome recurrent readout."""
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "experiments"))
from malecns_bge import BGESparseRecurrent, load_bge
from malecns_sentiment import GRAPH, mask_for, read_measured_graph


def test_training_embeddings_align_and_no_test_required():
    X, y = load_bge(
        root / "train.csv",
        root / "experiments/features/bge_small_train.npy",
        root / "experiments/features/bge_small_train_rows.csv",
    )
    assert X.shape == (511, 384)
    assert len(y) == 511 and set(y) == set(range(5))


def test_frozen_bge_sparse_recurrent_network_gradients():
    graph, _, _ = read_measured_graph(GRAPH, 32)
    mask, _ = mask_for(graph, 32, "biological", 42)
    net = BGESparseRecurrent(mask, input_dim=384, steps=6)
    x = torch.randn(7, 384)
    logits = net(x)
    assert logits.shape == (7, 5) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.arange(7) % 5).backward()
    assert torch.count_nonzero(net.weights.grad[net.mask == 0]) == 0
