"""Invariant tests for token-sequence RNN and LSTM comparators."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from malecns_sentiment import GRAPH, SparseConnectomeRNN, mask_for, read_measured_graph, tensorize
from rnn_lstm_sentiment import StandardSequenceClassifier, run_fold


def test_rnn_lstm_forward_and_padding_invariance():
    texts = ["Excellent film!", "Poor acting", "An enjoyable but uneven film indeed"]
    x, length = tensorize(texts, 32)
    padded = F.pad(x, (0, 7), value=0)
    for kind in ("rnn", "lstm"):
        model = StandardSequenceClassifier(kind, hidden_size=80)
        model.eval()
        logits = model(x, length)
        assert logits.shape == (3, 5)
        assert torch.isfinite(logits).all()
        assert torch.allclose(logits, model(padded, length), atol=1e-6)


def test_recurrent_gradients_and_parameter_counts():
    x, length = tensorize(["great film", "bad movie", "very average"], 24)
    labels = torch.tensor([4, 0, 2])
    counts = {}
    for kind in ("rnn", "lstm"):
        model = StandardSequenceClassifier(kind, hidden_size=80)
        loss = F.cross_entropy(model(x, length), labels)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        counts[kind] = sum(p.numel() for p in model.parameters())
    assert counts["lstm"] > counts["rnn"]


def test_outer_fold_predictions_only_training_labels():
    texts = ["not good", "good movie", "bad film", "great film", "very boring",
             "excellent story", "not great", "good cast", "pretty bad", "ok film"]
    X, lens = tensorize(texts, 16)
    labels = np.array([0, 4, 1, 4, 0, 4, 1, 3, 0, 2])
    tr = np.array([0, 1, 2, 3, 4, 5, 6, 9])
    va = np.array([7, 8])
    a, count = run_fold("rnn", None, X, lens, labels, tr, va,
                        seed=42, epochs=1, batch_size=4, lr=0.003,
                        device="cpu", dropout=0.25, hidden_size=32)
    flipped = labels.copy()
    flipped[va] = [4, 4]  # Changing held-out labels must not change any predictions.
    b, _ = run_fold("rnn", None, X, lens, flipped, tr, va,
                    seed=42, epochs=1, batch_size=4, lr=0.003,
                    device="cpu", dropout=0.25, hidden_size=32)
    assert a.shape == (2, 5) and count > 0
    assert np.array_equal(a, b)


def test_sparse_malecns_shared_path_unchanged():
    measured, _, _ = read_measured_graph(GRAPH, 80)
    mask, _ = mask_for(measured, 80, "biological", 42)
    model = SparseConnectomeRNN(mask)
    assert (model.mask.sum().item()) == 1296
