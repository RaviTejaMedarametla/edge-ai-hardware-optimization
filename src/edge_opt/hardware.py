from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn


@dataclass
class LayerEstimate:
    layer: str
    output_elements: int
    parameter_bytes: int
    activation_bytes: int
    macs: int


def _module_parameter_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters(recurse=False))


def _module_macs(module: nn.Module, output: torch.Tensor) -> int:
    if isinstance(module, nn.Conv2d):
        output_elements = output.numel()
        kernel_height, kernel_width = module.kernel_size
        groups = max(module.groups, 1)
        in_channels_per_group = module.in_channels // groups
        return int(output_elements * in_channels_per_group * kernel_height * kernel_width)

    if isinstance(module, nn.Linear):
        batch_size = output.shape[0] if output.ndim > 0 else 1
        return int(batch_size * module.in_features * module.out_features)

    return 0


def estimate_layerwise_stats(
    model: nn.Module,
    batch_size: int,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    activation_bytes_per_value: int | None = None,
) -> pd.DataFrame:
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("Model has no parameters to profile")

    activation_size = activation_bytes_per_value or first_parameter.element_size()
    tracked_modules: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            tracked_modules.append((name, module))

    if not tracked_modules:
        raise ValueError("No supported layers found for profiling")

    layer_rows: list[LayerEstimate] = []
    hooks = []

    def _make_hook(layer_name: str, layer_module: nn.Module):
        def _hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            output_tensor = output if isinstance(output, torch.Tensor) else output[0]
            output_elements = int(output_tensor.numel())
            layer_rows.append(
                LayerEstimate(
                    layer=layer_name,
                    output_elements=output_elements,
                    parameter_bytes=_module_parameter_bytes(layer_module),
                    activation_bytes=output_elements * activation_size,
                    macs=_module_macs(layer_module, output_tensor),
                )
            )

        return _hook

    for layer_name, layer_module in tracked_modules:
        hooks.append(layer_module.register_forward_hook(_make_hook(layer_name, layer_module)))

    try:
        device = first_parameter.device
        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            sample = torch.zeros((batch_size, *input_shape), device=device, dtype=first_parameter.dtype)
            model(sample)
        if model_was_training:
            model.train()
    finally:
        for hook in hooks:
            hook.remove()

    return pd.DataFrame([vars(row) for row in layer_rows])


def peak_activation_memory(
    model: nn.Module,
    batch_size: int,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    bytes_per_value: int | None = None,
) -> float:
    """Estimate activation-memory upper bound (MB) across profiled layers."""
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        return 0.0

    activation_size = bytes_per_value or first_parameter.element_size()
    device = first_parameter.device
    dtype = first_parameter.dtype
    activation_bytes: list[int] = []
    hooks = []

    def _record(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output_tensor = output if isinstance(output, torch.Tensor) else output[0]
        activation_bytes.append(int(output_tensor.numel()) * activation_size)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(_record))

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            sample = torch.zeros((batch_size, *input_shape), device=device, dtype=dtype)
            model(sample)
    finally:
        for hook in hooks:
            hook.remove()
        if model_was_training:
            model.train()

    return float(sum(activation_bytes) / (1024**2))


def summarize_hardware(
    layerwise_df: pd.DataFrame,
    latency_ms: float,
    memory_bandwidth_gbps: float,
    peak_compute_gmacs: float | None = None,
) -> dict[str, float | str]:
    total_bytes = float(layerwise_df["parameter_bytes"].sum() + layerwise_df["activation_bytes"].sum())
    total_macs = float(layerwise_df["macs"].sum())
    latency_s = max(latency_ms / 1000.0, 1e-9)
    achieved_bandwidth_gbps = (total_bytes / latency_s) / 1e9
    bandwidth_utilization = achieved_bandwidth_gbps / max(memory_bandwidth_gbps, 1e-9)
    achieved_gmacs = (total_macs / latency_s) / 1e9
    arithmetic_intensity = total_macs / max(total_bytes, 1.0)

    summary: dict[str, float | str] = {
        "estimated_total_bytes": total_bytes,
        "estimated_total_macs": total_macs,
        "achieved_bandwidth_gbps": achieved_bandwidth_gbps,
        "configured_memory_bandwidth_gbps": memory_bandwidth_gbps,
        "bandwidth_utilization": bandwidth_utilization,
        "achieved_gmacs": achieved_gmacs,
        "arithmetic_intensity_macs_per_byte": arithmetic_intensity,
    }

    if peak_compute_gmacs is not None:
        roofline_knee = peak_compute_gmacs / max(memory_bandwidth_gbps, 1e-9)
        bound = "memory-bound" if arithmetic_intensity < roofline_knee else "compute-bound"
        summary.update(
            {
                "configured_peak_compute_gmacs": peak_compute_gmacs,
                "roofline_knee_macs_per_byte": roofline_knee,
                "bound_regime": bound,
            }
        )

    return summary


def precision_tradeoff_table(sweep_df: pd.DataFrame) -> pd.DataFrame:
    grouped = sweep_df.groupby("precision", as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        memory_mb_mean=("memory_mb", "mean"),
        energy_proxy_j_mean=("energy_proxy_j", "mean"),
        accepted_ratio=("accepted", "mean"),
    )
    return grouped.sort_values("latency_ms_mean").reset_index(drop=True)


def save_hardware_artifacts(
    output_dir: Path,
    layerwise_df: pd.DataFrame,
    precision_df: pd.DataFrame,
    summary: dict[str, float | str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layerwise_df.to_csv(output_dir / "layerwise_breakdown.csv", index=False)
    precision_df.to_csv(output_dir / "precision_tradeoffs.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "hardware_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(layerwise_df["layer"], layerwise_df["activation_bytes"] / (1024**2), color="tab:orange")
    plt.ylabel("Activation Memory (MB)")
    plt.title("Layer-wise Activation Memory")
    plt.tight_layout()
    plt.savefig(output_dir / "layerwise_activation_memory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(layerwise_df["layer"], layerwise_df["macs"] / 1e6, color="tab:blue")
    plt.ylabel("MACs (Millions)")
    plt.title("Layer-wise Compute Estimate")
    plt.tight_layout()
    plt.savefig(output_dir / "layerwise_macs.png", dpi=180)
    plt.close()
