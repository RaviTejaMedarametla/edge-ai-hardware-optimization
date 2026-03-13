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


def _prune_conv2d(module: nn.Conv2d, keep_out: torch.Tensor, keep_in: torch.Tensor | None) -> nn.Conv2d:
    weight = module.weight.data[keep_out]
    if keep_in is not None:
        weight = weight[:, keep_in, :, :]
    bias = module.bias.data[keep_out] if module.bias is not None else None

    pruned = nn.Conv2d(
        in_channels=weight.shape[1],
        out_channels=weight.shape[0],
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=bias is not None,
        padding_mode=module.padding_mode,
    )
    with torch.no_grad():
        pruned.weight.copy_(weight)
        if bias is not None and pruned.bias is not None:
            pruned.bias.copy_(bias)
    return pruned


def _prune_linear_inputs(module: nn.Linear, keep_in: torch.Tensor | None) -> nn.Linear:
    if keep_in is None:
        return deepcopy(module)

    weight = module.weight.data
    if int(keep_in.max().item()) < module.in_features:
        selected = keep_in
    elif module.in_features % int(keep_in.numel()) == 0:
        # Handles flattened conv activations with evenly repeated channel blocks.
        repeat = module.in_features // int(keep_in.numel())
        offsets = torch.arange(repeat, device=keep_in.device)
        selected = (keep_in[:, None] * repeat + offsets[None, :]).reshape(-1)
    else:
        # Unknown mapping; keep layer unchanged rather than silently corrupting shape.
        return deepcopy(module)

    pruned = nn.Linear(
        in_features=int(selected.numel()),
        out_features=module.out_features,
        bias=module.bias is not None,
    )
    with torch.no_grad():
        pruned.weight.copy_(weight[:, selected])
        if module.bias is not None and pruned.bias is not None:
            pruned.bias.copy_(module.bias.data)
    return pruned


def structured_channel_prune(model: nn.Module, pruning_level: float) -> nn.Module:
    if not 0.0 <= pruning_level < 1.0:
        msg = "pruning_level must be in [0.0, 1.0)."
        raise ValueError(msg)

    pruned = deepcopy(model)
    pending_input_keep: torch.Tensor | None = None

    for name, module in list(pruned.named_modules()):
        if isinstance(module, nn.Conv2d):
            conv_scores = module.weight.data.abs().sum(dim=(1, 2, 3))
            keep_out = _topk_indices(conv_scores, pruning_level)
            replacement = _prune_conv2d(module, keep_out, pending_input_keep)
            parent_name, _, child_name = name.rpartition(".")
            parent = pruned.get_submodule(parent_name) if parent_name else pruned
            setattr(parent, child_name, replacement)
            pending_input_keep = keep_out

        elif isinstance(module, nn.Linear):
            replacement = _prune_linear_inputs(module, pending_input_keep)
            parent_name, _, child_name = name.rpartition(".")
            parent = pruned.get_submodule(parent_name) if parent_name else pruned
            setattr(parent, child_name, replacement)
            pending_input_keep = None

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
