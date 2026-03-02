from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader


def deployment_simulation(model: nn.Module, loader: DataLoader, cpu_frequency_scale: float, stream_items: int = 128) -> dict[str, float]:
    model.eval()
    batch_inputs, _ = next(iter(loader))
    latency_multiplier = 1.0 / max(cpu_frequency_scale, 1e-6)

    with torch.no_grad():
        start_batch = time.perf_counter()
        _ = model(batch_inputs)
        batch_time = (time.perf_counter() - start_batch) * latency_multiplier

        stream = batch_inputs[:stream_items]
        start_stream = time.perf_counter()
        for item in stream:
            _ = model(item.unsqueeze(0))
        stream_time = (time.perf_counter() - start_stream) * latency_multiplier

    return {
        "cpu_frequency_scale": cpu_frequency_scale,
        "latency_multiplier": latency_multiplier,
        "batch_latency_ms": batch_time * 1000.0,
        "batch_throughput_sps": batch_inputs.shape[0] / batch_time,
        "stream_avg_latency_ms": (stream_time / stream.shape[0]) * 1000.0,
        "stream_throughput_sps": stream.shape[0] / stream_time,
    }
