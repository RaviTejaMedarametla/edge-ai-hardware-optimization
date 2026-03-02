from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader


def to_fp16(model: nn.Module) -> nn.Module:
    fp16_model = deepcopy(model).half().eval()
    return fp16_model


def to_int8(model: nn.Module, calibration_loader: DataLoader, calibration_batches: int = 10) -> nn.Module:
    float_model = deepcopy(model).eval()
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    example_inputs, _ = next(iter(calibration_loader))
    prepared = prepare_fx(float_model, qconfig_mapping, example_inputs=(example_inputs,))

    with torch.no_grad():
        for index, (inputs, _) in enumerate(calibration_loader):
            _ = prepared(inputs)
            if index + 1 >= calibration_batches:
                break

    quantized = convert_fx(prepared)
    return quantized
