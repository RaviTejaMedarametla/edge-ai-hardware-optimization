from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils import benchmark
from torch.utils.data import DataLoader

from edge_opt.hardware import peak_activation_memory


@dataclass
class PerfMetrics:
    accuracy: float
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


def bootstrap_ci(data: np.ndarray, statistic=np.mean, n_resamples: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (mean, lower, upper)."""
    rng = np.random.default_rng()
    boot_stats = []
    n = len(data)
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        sample = data[indices]
        boot_stats.append(statistic(sample))
    boot_stats = np.array(boot_stats)
    mean = statistic(data)
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return mean, lower, upper


def evaluate_accuracy_with_bootstrap(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    n_resamples: int = 1000,
) -> tuple[float, float, float]:
    model.eval()
    all_correct = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if precision == "fp16":
                inputs = inputs.half()
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct = (pred == targets).cpu().numpy().astype(int)
            all_correct.extend(correct)
    all_correct = np.array(all_correct)
    mean, low, high = bootstrap_ci(all_correct, statistic=np.mean, n_resamples=n_resamples, ci=0.95)
    return mean, low, high


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
    benchmark_trials: int = 3,
    benchmark_warmup: int = 3,
) -> PerfMetrics:
    if len(loader) == 0:
        raise ValueError("No batches in loader; check dataset or split.")

    sample_batch, _ = next(iter(loader))
    sample_input = sample_batch.to(device)
    if precision == "fp16":
        sample_input = sample_input.half()

    accuracy, ci_low, ci_high = evaluate_accuracy_with_bootstrap(
        model, loader, device, precision, n_resamples=1000
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
    batch_size = sample_input.shape[0]
    input_shape = sample_input.shape[1:]
    peak_mem = peak_activation_memory(model, batch_size, input_shape)
    runtime_memory = param_memory + peak_mem

    energy_proxy = (latency / 1000.0) * power_watts

    return PerfMetrics(
        accuracy=accuracy,
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
