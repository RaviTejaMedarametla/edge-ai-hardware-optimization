from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils import benchmark
from torch.utils.data import DataLoader


@dataclass
class PerfMetrics:
    accuracy: float
    accuracy_std: float
    accuracy_ci95_low: float
    accuracy_ci95_high: float
    latency_ms: float
    latency_median_ms: float
    latency_std_ms: float
    latency_p95_ms: float
    throughput_sps: float
    model_memory_mb: float
    memory_mb: float
    estimated_runtime_memory_mb: float
    energy_proxy_j: float
    energy_proxy_note: str


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


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


def evaluate_accuracy_distribution(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    trials: int,
) -> tuple[float, float, float, float]:
    accuracies = [evaluate_accuracy(model, loader, device, precision=precision) for _ in range(trials)]
    tensor = torch.tensor(accuracies, dtype=torch.float32)
    mean = float(tensor.mean())
    std = float(tensor.std(unbiased=False))
    ci_half_width = float(1.96 * (std / max(trials**0.5, 1.0)))
    return mean, std, max(0.0, mean - ci_half_width), min(1.0, mean + ci_half_width)


def measure_latency(model: nn.Module, sample_input: torch.Tensor, device: torch.device, num_runs: int = 100, warmup: int = 3) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
        _sync_device(device)

        try:
            timer = benchmark.Timer(stmt="model(sample_input)", globals={"model": model, "sample_input": sample_input})
            result = timer.blocked_autorange(min_run_time=max(0.1, num_runs * 0.001))
            return result.median * 1000.0
        except Exception:
            start = time.perf_counter()
            for _ in range(num_runs):
                _ = model(sample_input)
            _sync_device(device)
            elapsed = time.perf_counter() - start
            return (elapsed / num_runs) * 1000.0


def measure_latency_distribution(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    repeats: int = 5,
    num_runs: int = 100,
    warmup: int = 3,
) -> tuple[float, float, float, float]:
    latencies = [measure_latency(model, sample_input, device=device, num_runs=num_runs, warmup=warmup) for _ in range(repeats)]
    latency_tensor = torch.tensor(latencies, dtype=torch.float32)
    return (
        float(latency_tensor.mean()),
        float(torch.median(latency_tensor)),
        float(latency_tensor.std(unbiased=False)),
        float(torch.quantile(latency_tensor, 0.95)),
    )


def model_memory_mb(model: nn.Module) -> float:
    """Compute model parameter memory from the state dict only."""
    total_bytes = 0
    for tensor in model.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def estimated_runtime_memory_mb(parameter_memory_mb: float) -> float:
    """Estimate runtime memory including activations as ~1.5x parameter memory."""
    return parameter_memory_mb * 1.5


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
    benchmark_trials: int = 3,
    benchmark_warmup: int = 3,
) -> PerfMetrics:
    if len(loader) == 0:
        raise ValueError("No batches in loader; check dataset or split.")

    sample_batch, _ = next(iter(loader))
    sample_input = sample_batch.to(device)
    if precision == "fp16":
        sample_input = sample_input.half()

    accuracy, accuracy_std, ci_low, ci_high = evaluate_accuracy_distribution(
        model,
        loader,
        device,
        precision=precision,
        trials=benchmark_trials,
    )
    latency_mean, latency_median, latency_std, latency_p95 = measure_latency_distribution(
        model,
        sample_input,
        device=device,
        repeats=benchmark_repeats,
        warmup=benchmark_warmup,
    )
    latency = latency_mean * latency_multiplier
    throughput = sample_input.shape[0] / (latency / 1000.0)
    param_memory = model_memory_mb(model)
    runtime_memory = estimated_runtime_memory_mb(param_memory)
    energy_proxy = (latency / 1000.0) * power_watts

    return PerfMetrics(
        accuracy=accuracy,
        accuracy_std=accuracy_std,
        accuracy_ci95_low=ci_low,
        accuracy_ci95_high=ci_high,
        latency_ms=latency,
        latency_median_ms=latency_median * latency_multiplier,
        latency_std_ms=latency_std * latency_multiplier,
        latency_p95_ms=latency_p95 * latency_multiplier,
        throughput_sps=throughput,
        model_memory_mb=param_memory,
        memory_mb=param_memory,
        estimated_runtime_memory_mb=runtime_memory,
        energy_proxy_j=energy_proxy,
        energy_proxy_note="Proxy metric computed as power_watts × latency_s (not measured on-device power)",
    )
