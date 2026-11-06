"""Optional differentiable objectives for ordered 0..4 sentiment labels.

All losses operate on a training minibatch. Never provide evaluation labels.
These objectives are experimental; validation, not geometric appearance, decides merit.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def adjacent_error_cost(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Expected probability assigned to wrong *adjacent* labels.

    Ordinary CE is still required: this regularizer intentionally penalizes
    nearby errors more than CE alone, without forgiving distant errors.
    """
    if logits.ndim != 2 or logits.shape[1] != 5 or targets.ndim != 1:
        raise ValueError("Expected (batch,5) logits and (batch,) labels")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("Mismatched batch sizes")
    probs = logits.softmax(dim=1)
    classes = torch.arange(5, device=logits.device)
    cost = (classes.unsqueeze(0) - targets.long().unsqueeze(1)).abs().eq(1)
    return (probs * cost.to(probs.dtype)).sum(dim=1).mean()


def adjacent_triplet_margin(
    embeddings: torch.Tensor, targets: torch.Tensor,
    boundary_margins=(0.22, 0.16, 0.16, 0.22),
) -> torch.Tensor:
    """Require positive pairs to be closer than nearest adjacent-class negatives.

    Only anchors with a same-label peer and an adjacent-class peer contribute;
    missing classes or singleton classes produce zero instead of bogus gradients.
    The four margins correspond to boundaries 0-1, 1-2, 2-3, 3-4.
    """
    if len(boundary_margins) != 4 or embeddings.shape[0] != len(targets):
        raise ValueError("Four margins and aligned embeddings/labels required")
    z = F.normalize(embeddings, p=2, dim=1)
    sim = z @ z.T
    n = len(targets)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    positive = (targets[:, None] == targets[None, :]) & ~eye
    adjacent = (targets[:, None] - targets[None, :]).abs() == 1
    valid = positive.any(1) & adjacent.any(1)
    if not valid.any():
        return z.sum() * 0
    # Farthest positive / closest adjacent negative (cosine similarity).
    pos_similarity = sim.masked_fill(~positive, 2.).min(dim=1).values
    neg_similarity, neg_indices = sim.masked_fill(~adjacent, -2.).max(dim=1)
    # Apply the valid-anchor mask BEFORE looking up boundary margins.
    # Invalid anchors may have no adjacent-class candidate, making their
    # argmax fallback an unrelated label (even boundary index 4).
    boundary = torch.minimum(targets[valid], targets[neg_indices[valid]]).long()
    margins = z.new_tensor(boundary_margins)[boundary]
    return F.relu(margins + neg_similarity[valid] - pos_similarity[valid]).mean()


def adjacent_prototype_margin(
    embeddings: torch.Tensor, targets: torch.Tensor, margin: float = 0.30
) -> torch.Tensor:
    """Separate adjacent *minibatch* class centroids by cosine distance.

    Within-class cohesion is encouraged by CE and triplet terms. This term
    alone cannot create reliable clusters on unseen reviews.
    """
    z = F.normalize(embeddings, p=2, dim=1)
    centers = []
    for cls in range(5):
        mask = targets.eq(cls)
        centers.append(F.normalize(z[mask].mean(0), dim=0) if mask.any() else None)
    penalties = []
    for cls in range(4):
        if centers[cls] is not None and centers[cls + 1] is not None:
            distance = 1. - (centers[cls] * centers[cls + 1]).sum()
            penalties.append(F.relu(z.new_tensor(margin) - distance))
    return torch.stack(penalties).mean() if penalties else z.sum() * 0
