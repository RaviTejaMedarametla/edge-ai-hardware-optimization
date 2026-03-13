from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader


def _topk_indices(channel_scores: torch.Tensor, pruning_level: float) -> torch.Tensor:
    total = channel_scores.numel()
    keep = max(1, int(round(total * (1.0 - pruning_level))))
    return torch.topk(channel_scores, keep, largest=True).indices.sort().values


def structured_channel_prune(model: nn.Module, pruning_level: float) -> nn.Module:
    if not 0.0 <= pruning_level < 1.0:
        raise ValueError("pruning_level must be in [0.0, 1.0).")

    pruned = deepcopy(model)
    conv_layers = [(name, module) for name, module in pruned.named_modules() if isinstance(module, nn.Conv2d)]
    linear_layers = [(name, module) for name, module in pruned.named_modules() if isinstance(module, nn.Linear)]
    if not conv_layers or not linear_layers:
        return pruned

    conv_name, conv = conv_layers[-1]
    classifier_name, classifier = linear_layers[-1]

    scores = conv.weight.data.abs().sum(dim=(1, 2, 3))
    keep = _topk_indices(scores, pruning_level)

    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=len(keep),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[keep])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias[keep])

    in_features_per_channel = classifier.in_features // conv.out_channels
    new_classifier = nn.Linear(in_features_per_channel * len(keep), classifier.out_features, bias=classifier.bias is not None)

    with torch.no_grad():
        weight_view = classifier.weight.view(classifier.out_features, conv.out_channels, in_features_per_channel)
        new_classifier.weight.copy_(weight_view[:, keep, :].reshape(classifier.out_features, -1))
        if classifier.bias is not None:
            new_classifier.bias.copy_(classifier.bias)

    parent_name = conv_name.rsplit('.', 1)[0] if '.' in conv_name else ''
    conv_attr = conv_name.split('.')[-1]
    parent = pruned.get_submodule(parent_name) if parent_name else pruned
    setattr(parent, conv_attr, new_conv)

    classifier_parent_name = classifier_name.rsplit('.', 1)[0] if '.' in classifier_name else ''
    classifier_attr = classifier_name.split('.')[-1]
    classifier_parent = pruned.get_submodule(classifier_parent_name) if classifier_parent_name else pruned
    setattr(classifier_parent, classifier_attr, new_classifier)

    return pruned


def prune_and_finetune(
    model: nn.Module,
    pruning_level: float,
    fine_tune_epochs: int,
    train_loader: DataLoader,
    train_one_epoch: Callable[[nn.Module, DataLoader], nn.Module],
) -> nn.Module:
    pruned = structured_channel_prune(model, pruning_level)
    for _ in range(fine_tune_epochs):
        pruned = train_one_epoch(pruned, train_loader)
    return pruned
