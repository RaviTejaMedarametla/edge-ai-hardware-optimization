from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class PerfMetrics:
    accuracy: float
    latency_ms: float
    latency_std_ms: float
    latency_p95_ms: float
    throughput_sps: float
    memory_mb: float
    energy_proxy_j: float


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, precision: str = "fp32") -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if precision == "fp16":
                inputs = inputs.half()
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
    return correct / total


def measure_latency(model: nn.Module, sample_input: torch.Tensor, num_runs: int = 100, warmup: int = 10) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(sample_input)
        elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000.0




def measure_latency_distribution(model: nn.Module, sample_input: torch.Tensor, repeats: int = 5, num_runs: int = 100, warmup: int = 10) -> tuple[float, float, float]:
    latencies = [measure_latency(model, sample_input, num_runs=num_runs, warmup=warmup) for _ in range(repeats)]
    latency_tensor = torch.tensor(latencies, dtype=torch.float32)
    return float(latency_tensor.mean()), float(latency_tensor.std(unbiased=False)), float(torch.quantile(latency_tensor, 0.95))

def model_memory_mb(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def memory_violations(memory_mb: float, budgets_mb: list[float]) -> dict[str, bool]:
    return {f"violates_{budget}mb": memory_mb > budget for budget in budgets_mb}


def collect_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    power_watts: float,
    precision: str,
    latency_multiplier: float = 1.0,
    benchmark_repeats: int = 5,
) -> PerfMetrics:
    sample_batch, _ = next(iter(loader))
    sample_input = sample_batch.to(device)
    if precision == "fp16":
        sample_input = sample_input.half()

    accuracy = evaluate_accuracy(model, loader, device, precision=precision)
    latency_mean, latency_std, latency_p95 = measure_latency_distribution(model, sample_input, repeats=benchmark_repeats)
    latency = latency_mean * latency_multiplier
    throughput = sample_input.shape[0] / (latency / 1000.0)
    memory = model_memory_mb(model)
    energy_proxy = (latency / 1000.0) * power_watts

    return PerfMetrics(
        accuracy=accuracy,
        latency_ms=latency,
        latency_std_ms=latency_std * latency_multiplier,
        latency_p95_ms=latency_p95 * latency_multiplier,
        throughput_sps=throughput,
        memory_mb=memory,
        energy_proxy_j=energy_proxy,
    )
