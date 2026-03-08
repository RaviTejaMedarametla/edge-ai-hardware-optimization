from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from torch import nn


@dataclass
class LayerEstimate:
    layer: str
    output_elements: int
    parameter_bytes: int
    activation_bytes: int
    macs: int


def _conv2d_output_shape(height: int, width: int, kernel: int, padding: int, stride: int = 1) -> tuple[int, int]:
    out_h = (height + (2 * padding) - kernel) // stride + 1
    out_w = (width + (2 * padding) - kernel) // stride + 1
    return out_h, out_w


def estimate_layerwise_stats(
    model: nn.Module,
    batch_size: int,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    activation_bytes_per_value: int | None = None,
) -> pd.DataFrame:
    _, height, width = input_shape

    conv1 = model.conv1
    conv2 = model.conv2
    linear = model.classifier

    parameter_bytes_per_value = conv1.weight.element_size()
    activation_bytes_per_value = activation_bytes_per_value or parameter_bytes_per_value

    h1, w1 = _conv2d_output_shape(height, width, kernel=3, padding=1)
    h1_pool, w1_pool = h1 // 2, w1 // 2
    conv1_elements = batch_size * conv1.out_channels * h1 * w1
    conv1_macs = batch_size * conv1.out_channels * h1 * w1 * conv1.in_channels * 3 * 3

    h2, w2 = _conv2d_output_shape(h1_pool, w1_pool, kernel=3, padding=1)
    conv2_elements = batch_size * conv2.out_channels * h2 * w2
    conv2_macs = batch_size * conv2.out_channels * h2 * w2 * conv2.in_channels * 3 * 3

    linear_elements = batch_size * linear.out_features
    linear_macs = batch_size * linear.in_features * linear.out_features

    rows = [
        LayerEstimate(
            layer="conv1",
            output_elements=conv1_elements,
            parameter_bytes=(conv1.weight.numel() + conv1.bias.numel()) * parameter_bytes_per_value,
            activation_bytes=conv1_elements * activation_bytes_per_value,
            macs=conv1_macs,
        ),
        LayerEstimate(
            layer="conv2",
            output_elements=conv2_elements,
            parameter_bytes=(conv2.weight.numel() + conv2.bias.numel()) * parameter_bytes_per_value,
            activation_bytes=conv2_elements * activation_bytes_per_value,
            macs=conv2_macs,
        ),
        LayerEstimate(
            layer="classifier",
            output_elements=linear_elements,
            parameter_bytes=(linear.weight.numel() + linear.bias.numel()) * parameter_bytes_per_value,
            activation_bytes=linear_elements * activation_bytes_per_value,
            macs=linear_macs,
        ),
    ]
    return pd.DataFrame([vars(row) for row in rows])


def summarize_hardware(
    layerwise_df: pd.DataFrame,
    latency_ms: float,
    memory_bandwidth_gbps: float,
) -> dict[str, float]:
    total_bytes = float(layerwise_df["parameter_bytes"].sum() + layerwise_df["activation_bytes"].sum())
    total_macs = float(layerwise_df["macs"].sum())
    latency_s = max(latency_ms / 1000.0, 1e-9)
    achieved_bandwidth_gbps = (total_bytes / latency_s) / 1e9
    bandwidth_utilization = achieved_bandwidth_gbps / max(memory_bandwidth_gbps, 1e-9)
    achieved_gmacs = (total_macs / latency_s) / 1e9
    return {
        "estimated_total_bytes": total_bytes,
        "estimated_total_macs": total_macs,
        "achieved_bandwidth_gbps": achieved_bandwidth_gbps,
        "configured_memory_bandwidth_gbps": memory_bandwidth_gbps,
        "bandwidth_utilization": bandwidth_utilization,
        "achieved_gmacs": achieved_gmacs,
    }


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
    summary: dict[str, float],
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
