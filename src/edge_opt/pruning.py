from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from edge_opt.model import SmallCNN


def _topk_indices(channel_scores: torch.Tensor, pruning_level: float) -> torch.Tensor:
    total = channel_scores.numel()
    keep = max(1, int(round(total * (1.0 - pruning_level))))
    return torch.topk(channel_scores, keep, largest=True).indices.sort().values


def structured_channel_prune(model: SmallCNN, pruning_level: float) -> SmallCNN:
    if not 0.0 <= pruning_level < 1.0:
        msg = "pruning_level must be in [0.0, 1.0)."
        raise ValueError(msg)

    conv1_scores = model.conv1.weight.data.abs().sum(dim=(1, 2, 3))
    keep1 = _topk_indices(conv1_scores, pruning_level)

    conv2_scores = model.conv2.weight.data.abs().sum(dim=(1, 2, 3))
    keep2 = _topk_indices(conv2_scores, pruning_level)

    pruned = SmallCNN(conv1_channels=len(keep1), conv2_channels=len(keep2), num_classes=model.classifier.out_features)

    with torch.no_grad():
        pruned.conv1.weight.copy_(model.conv1.weight[keep1])
        pruned.conv1.bias.copy_(model.conv1.bias[keep1])

        conv2_w = model.conv2.weight[keep2][:, keep1, :, :]
        pruned.conv2.weight.copy_(conv2_w)
        pruned.conv2.bias.copy_(model.conv2.bias[keep2])

        pruned.classifier.weight.copy_(model.classifier.weight[:, keep2])
        pruned.classifier.bias.copy_(model.classifier.bias)

    return pruned


def prune_and_finetune(
    model: SmallCNN,
    pruning_level: float,
    fine_tune_epochs: int,
    train_loader: DataLoader,
    train_one_epoch: Callable[[SmallCNN, DataLoader], SmallCNN],
) -> SmallCNN:
    pruned = structured_channel_prune(model, pruning_level)
    for _ in range(fine_tune_epochs):
        pruned = train_one_epoch(pruned, train_loader)
    return pruned
