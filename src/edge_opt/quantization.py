from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


class FakeQuantLinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.linear = deepcopy(base)
        self.scale: torch.Tensor | None = None
        self.zero_point: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is None or self.zero_point is None:
            return self.linear(x)
        x_q = torch.round(x / self.scale) + self.zero_point
        x_q = torch.clamp(x_q, -128, 127)
        x_dq = (x_q - self.zero_point) * self.scale
        return self.linear(x_dq)


class FakeQuantConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d):
        super().__init__()
        self.conv = deepcopy(base)
        self.scale: torch.Tensor | None = None
        self.zero_point: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is None or self.zero_point is None:
            return self.conv(x)
        x_q = torch.round(x / self.scale) + self.zero_point
        x_q = torch.clamp(x_q, -128, 127)
        x_dq = (x_q - self.zero_point) * self.scale
        return self.conv(x_dq)


def _replace_with_fake_quant(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, FakeQuantConv2d(child))
        else:
            _replace_with_fake_quant(child)


def prepare_fake_quant(model: nn.Module) -> nn.Module:
    prepared = deepcopy(model).eval().to("cpu")
    _replace_with_fake_quant(prepared)
    return prepared


def calibrate_fake_quant(model: nn.Module, calibration_loader: DataLoader, num_batches: int = 10) -> nn.Module:
    stats: dict[nn.Module, list[torch.Tensor]] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(mod: nn.Module):
        def _hook(_: nn.Module, __: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            tensor = out if isinstance(out, torch.Tensor) else out[0]
            stats.setdefault(mod, []).append(tensor.detach().cpu())

        return _hook

    for mod in model.modules():
        if isinstance(mod, (FakeQuantLinear, FakeQuantConv2d)):
            hooks.append(mod.register_forward_hook(_make_hook(mod)))

    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            if i >= num_batches:
                break
            _ = model(inputs.cpu())

    for hook in hooks:
        hook.remove()

    for mod, activations in stats.items():
        if not activations:
            continue
        merged = torch.cat([a.flatten() for a in activations])
        amax = torch.max(torch.abs(merged))
        scale = torch.clamp(amax / 127.0, min=1e-8)
        mod.scale = scale
        mod.zero_point = torch.tensor(0.0)

    return model


def to_fp16(model: nn.Module) -> nn.Module:
    fp16_model = deepcopy(model).half().eval()
    return fp16_model


def to_int8(
    model: nn.Module,
    calibration_loader: DataLoader,
    calibration_batches: int = 10,
    backend: str | None = None,
    metadata_path: str | Path | None = None,
) -> nn.Module:
    _ = backend  # compatibility placeholder
    quantized = prepare_fake_quant(model)
    quantized = calibrate_fake_quant(quantized, calibration_loader, num_batches=calibration_batches)

    if metadata_path is not None:
        metadata = {
            "backend": "fake-quant",
            "calibration_batches": calibration_batches,
            "quantized": True,
            "layers": {},
        }
        for name, mod in quantized.named_modules():
            if isinstance(mod, (FakeQuantLinear, FakeQuantConv2d)):
                metadata["layers"][name] = {
                    "scale": float(mod.scale.item()) if mod.scale is not None else None,
                    "zero_point": int(mod.zero_point.item()) if mod.zero_point is not None else None,
                }

        path = Path(metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return quantized
