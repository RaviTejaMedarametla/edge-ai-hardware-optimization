from __future__ import annotations

import json
import warnings
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader


def _default_backend() -> str:
    return "qnnpack" if "arm" in torch.backends.quantized.engine.lower() else "fbgemm"


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
    backend_name = backend or _default_backend()
    float_model = deepcopy(model).eval().to("cpu")

    try:
        qconfig_mapping = get_default_qconfig_mapping(backend_name)
        example_inputs, _ = next(iter(calibration_loader))
        prepared = prepare_fx(float_model, qconfig_mapping, example_inputs=(example_inputs.cpu(),))

        with torch.no_grad():
            for index, (inputs, _) in enumerate(calibration_loader):
                _ = prepared(inputs.cpu())
                if index + 1 >= calibration_batches:
                    break

        quantized = convert_fx(prepared)
    except Exception as exc:
        warnings.warn(f"INT8 quantization backend '{backend_name}' failed ({exc}); using CPU float model.", stacklevel=2)
        quantized = float_model

    if metadata_path is not None:
        metadata = {
            "backend": backend_name,
            "calibration_batches": calibration_batches,
            "quantized": quantized is not float_model,
            "modules": {},
        }
        for name, module in quantized.named_modules():
            scale = getattr(module, "scale", None)
            zero_point = getattr(module, "zero_point", None)
            metadata["modules"][name or "root"] = {
                "type": module.__class__.__name__,
                "has_scale": scale is not None,
                "has_zero_point": zero_point is not None,
                "scale": float(scale) if scale is not None else None,
                "zero_point": int(zero_point) if zero_point is not None else None,
            }

        path = Path(metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return quantized
