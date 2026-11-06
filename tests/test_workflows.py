"""Small, data-free checks for the new training and annotation workflow."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from ordinal_losses import adjacent_error_cost, adjacent_triplet_margin, adjacent_prototype_margin
from active_learning_loop import load_features, query_batch, human_queue
from hard_negative_review import export_approved


def test_adjacent_loss_gradients_and_constraints():
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    logits = torch.randn(10, 5, requires_grad=True)
    z = torch.randn(10, 12, requires_grad=True)
    loss = adjacent_error_cost(logits, y) + adjacent_triplet_margin(z, y) + adjacent_prototype_margin(z, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(z.grad).all()
    assert adjacent_triplet_margin(z[:2], y[:2]).item() == 0.



def test_triplet_mask_handles_one_invalid_anchor_in_batch_of_24():
    # Reproduces the MPS crash: 24 margin candidates but only 23 valid triplets.
    y = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 8 + [4])
    z = torch.randn(len(y), 16, requires_grad=True)
    loss = adjacent_triplet_margin(z, y)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_triplet_mask_handles_nonadjacent_invalid_class_four():
    # Class 4 has positives but no class-3 neighbors. The invalid argmax
    # must never be used to index a 0..3 boundary margin.
    y = torch.tensor([4, 4, 2, 2, 1, 1])
    z = torch.randn(len(y), 16, requires_grad=True)
    loss = adjacent_triplet_margin(z, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()

def test_adjacent_boundary_penalty_is_high_for_adjacent_probabilities():
    y = torch.tensor([0])
    near = torch.tensor([[0., 5., -9., -9., -9.]])
    far = torch.tensor([[0., -9., -9., -9., 5.]])
    assert adjacent_error_cost(near, y) > adjacent_error_cost(far, y)


def test_pool_query_never_needs_gold_labels():
    rng = np.random.default_rng(3)
    p = np.array([[.48,.50,.01,.01,0], [.01,.01,.20,.48,.30], [.95,.01,.01,.01,.02]])
    X = np.eye(3)
    chosen = query_batch(p, X, [0,1,2], [], 1, "adjacent", rng)
    assert len(chosen) == 1 and chosen[0] == 0


def test_alignment_requires_phrase_ids(tmp_path):
    frame = pd.DataFrame({"PhraseId":[1,2], "Phrase":["bad","good"]})
    frame.to_csv(tmp_path/"input.csv",index=False)
    np.save(tmp_path/"x.npy",np.eye(2))
    pd.DataFrame({"PhraseId":[2,1]}).to_csv(tmp_path/"rows.csv",index=False)
    with pytest.raises(ValueError, match="order"):
        load_features(tmp_path/"input.csv",tmp_path/"x.npy",tmp_path/"rows.csv")


def test_human_queue_does_not_require_gold(tmp_path):
    frame = pd.DataFrame({"PhraseId":range(6),"Phrase":[f"sentence {i}" for i in range(6)]})
    queue, status = human_queue(frame,np.eye(6),tmp_path/"not_present.csv",3,"adjacent",42,1.)
    assert len(queue)==3 and (queue.Sentiment == "").all()
    assert status["labeled"]==0


def test_review_export_requires_verified_correct_labels(tmp_path):
    train=pd.DataFrame({"PhraseId":[1,2,3],"Sentiment":[0,0,1]})
    train.to_csv(tmp_path/"train.csv",index=False)
    aug=pd.DataFrame([{"PhraseId":1,"Sentiment":0,"Phrase":"Awful","augmented_text":"A truly awful film.",
                       "approved":True,"label_verified":True}])
    pairs=pd.DataFrame([{"anchor_id":1,"positive_id":2,"negative_id":3,
                         "anchor_label":0,"positive_label":0,"negative_label":1,
                         "approved":True,"labels_verified":True,"intensity_difference_verified":True}])
    aug.to_csv(tmp_path/"aug.csv",index=False);pairs.to_csv(tmp_path/"pairs.csv",index=False)
    assert export_approved(tmp_path/"aug.csv",tmp_path/"pairs.csv",tmp_path/"out",tmp_path/"train.csv")==(1,1)
    pairs["negative_label"]=2
    pairs.to_csv(tmp_path/"pairs.csv",index=False)
    with pytest.raises(ValueError,match="mismatch"):
        export_approved(tmp_path/"aug.csv",tmp_path/"pairs.csv",tmp_path/"out",tmp_path/"train.csv")


def test_hybrid_acquisition_has_no_unlabeled_truth_access():
    rng = np.random.default_rng(91)
    p = np.array([[.49, .47, .02, .01, .01],
                  [.01, .02, .47, .48, .02],
                  [.02, .02, .02, .47, .47],
                  [.9, .025, .025, .025, .025]])
    x = np.eye(9)
    pool = np.arange(4)
    labeled = np.arange(4, 9)
    known_labels = np.arange(5)
    for policy in ("hybrid", "balanced_hybrid"):
        picked = query_batch(p, x, pool, labeled, 2, policy,
                             rng, labeled_labels=known_labels)
        assert len(picked) == 2 and len(set(picked)) == 2
        assert set(picked).issubset(set(pool))


def test_balanced_hybrid_rejects_missing_training_labels():
    p = np.tile(np.array([[.2, .2, .2, .2, .2]]), (3, 1))
    with pytest.raises(ValueError, match="requires verified labels"):
        query_batch(p, np.eye(6), [0, 1, 2], [3, 4, 5],
                    2, "balanced_hybrid", np.random.default_rng(0))


def test_custom_adjacent_margins_have_finite_gradients():
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    z = torch.randn(10, 16, requires_grad=True)
    for margins in ((.12, .10, .10, .12), (.32, .24, .24, .32)):
        value = adjacent_triplet_margin(z, y, boundary_margins=margins)
        assert torch.isfinite(value)
        value.backward(retain_graph=True)
    assert torch.isfinite(z.grad).all()
