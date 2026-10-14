"""Binary mask scores used by the occupancy debug and visualization scripts."""

import torch


def occupancy_scores(logits, targets, threshold=0.5):
    """Return per-sample IoU/Dice from [1, 1, H, W] logits at ``threshold``.

    Both masks empty counts as perfect overlap; callers aggregate intersections
    and cell counts before computing a dataset-level score.
    """
    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[:2] != (1, 1):
        raise ValueError(f"expected matching [1, 1, H, W] logits/targets, got {logits.shape} and {targets.shape}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    with torch.no_grad():
        predicted = logits.detach().sigmoid() >= threshold
        target = targets.detach() >= 0.5
        intersection = int((predicted & target).sum().item())
        target_cells = int(target.sum().item())
        predicted_cells = int(predicted.sum().item())
    union = target_cells + predicted_cells - intersection
    total = target_cells + predicted_cells
    return dict(
        intersection=intersection,
        union=union,
        target_cells=target_cells,
        predicted_cells=predicted_cells,
        iou=intersection / union if union else 1.0,
        dice=2 * intersection / total if total else 1.0,
    )
