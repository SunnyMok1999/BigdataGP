"""Tests for the measured-graph recurrent classifier (no user test.csv access)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from malecns_sentiment import (
    GRAPH, SparseConnectomeRNN, directed_rewire, mask_for,
    read_measured_graph, tensorize, tokenize,
)


def test_real_malecns_graph_and_subsets():
    measured, neuron_ids, digest = read_measured_graph(GRAPH, 80)
    assert len(neuron_ids) == 80
    assert len(measured) == 1296
    assert len(digest) == 64
    assert max(max(s, d) for s, d in measured) < 80
    for n in (32, 64):
        subset, subset_ids, _ = read_measured_graph(GRAPH, n)
        assert len(subset_ids) == n and len(subset) > 0


def test_degree_preservation_and_mask_orientation():
    graph, _, _ = read_measured_graph(GRAPH, 80)
    rewired = directed_rewire(graph, seed=123)
    assert rewired != graph  # Rewiring must actually change the topology.
    assert len(rewired) == len(graph)
    for n in range(80):
        assert sum(s == n for s, d in rewired) == sum(s == n for s, d in graph)
        assert sum(d == n for s, d in rewired) == sum(d == n for s, d in graph)
    for kind in ("biological", "degree_rewired", "edge_random"):
        mask, edges = mask_for(graph, 80, kind, 123)
        assert mask.sum() == len(graph)
        assert all(mask[t, s] == 1 for s, t in edges)


def test_padding_and_masked_parameter_gradients():
    tokens, lens = tensorize(["an excellent film", "awful", "not very good"], 16)
    graph, _, _ = read_measured_graph(GRAPH, 32)
    mask, _ = mask_for(graph, 32, "biological", 1)
    model = SparseConnectomeRNN(mask)
    model.eval()
    logits = model(tokens, lens)
    assert logits.shape == (3, 5) and torch.isfinite(logits).all()
    padded = torch.nn.functional.pad(tokens, (0, 5), value=0)
    assert torch.allclose(logits, model(padded, lens), atol=1e-6)
    model.train()
    torch.nn.functional.cross_entropy(model(tokens, lens), torch.tensor([4, 0, 1])).backward()
    assert torch.isfinite(model.recurrent.grad).all()
    assert torch.count_nonzero(model.recurrent.grad[model.mask == 0]) == 0


def test_tokenizer_reproducibility():
    assert tokenize("Great! Loved it.") == tokenize("great! loved it.")
    assert len(tokenize("")) == 1
