from __future__ import annotations

import random
import warnings
from typing import Type

import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, conv1_channels: int = 16, conv2_channels: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(conv2_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


_MODEL_REGISTRY: dict[str, Type[nn.Module]] = {
    "SmallCNN": SmallCNN,
}


def get_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)


def register_model(name: str, model_cls: Type[nn.Module]) -> None:
    _MODEL_REGISTRY[name] = model_cls


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", stacklevel=2)
        return torch.device("cpu")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS requested but unavailable; falling back to CPU.", stacklevel=2)
        return torch.device("cpu")

    warnings.warn(f"Unknown device '{device_name}'; falling back to CPU.", stacklevel=2)
    return torch.device("cpu")


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
