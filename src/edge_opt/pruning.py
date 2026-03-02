from __future__ import annotations

import torch

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

    pruned = SmallCNN(conv1_channels=len(keep1), conv2_channels=len(keep2))

    with torch.no_grad():
        pruned.conv1.weight.copy_(model.conv1.weight[keep1])
        pruned.conv1.bias.copy_(model.conv1.bias[keep1])

        conv2_w = model.conv2.weight[keep2][:, keep1, :, :]
        pruned.conv2.weight.copy_(conv2_w)
        pruned.conv2.bias.copy_(model.conv2.bias[keep2])

        features_per_channel = 7 * 7
        fc_indices = []
        for channel in keep2.tolist():
            start = channel * features_per_channel
            fc_indices.extend(range(start, start + features_per_channel))
        fc_idx = torch.tensor(fc_indices, dtype=torch.long)

        pruned.classifier.weight.copy_(model.classifier.weight[:, fc_idx])
        pruned.classifier.bias.copy_(model.classifier.bias)

    return pruned
