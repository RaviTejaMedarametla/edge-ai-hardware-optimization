from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from edge_opt.metrics import PerfMetrics, collect_metrics, memory_violations
from edge_opt.pruning import structured_channel_prune
from edge_opt.quantization import to_fp16, to_int8


def train_model(model: nn.Module, train_loader: DataLoader, epochs: int, learning_rate: float, device: torch.device) -> nn.Module:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model


def precision_variant(model: nn.Module, precision: str, calibration_loader: DataLoader, calibration_batches: int) -> tuple[nn.Module, str]:
    if precision == "fp32":
        return deepcopy(model).eval(), "fp32"
    if precision == "fp16":
        return to_fp16(model), "fp16"
    if precision == "int8":
        return to_int8(model, calibration_loader, calibration_batches=calibration_batches), "fp32"
    msg = f"Unsupported precision '{precision}'"
    raise ValueError(msg)


def run_sweep(
    base_model: nn.Module,
    val_loader: DataLoader,
    calibration_loader: DataLoader,
    device: torch.device,
    pruning_levels: list[float],
    precisions: list[str],
    power_watts: float,
    calibration_batches: int,
    memory_budgets_mb: list[float],
    active_memory_budget_mb: float,
    latency_multiplier: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for pruning in pruning_levels:
        candidate = structured_channel_prune(base_model, pruning)
        for precision in precisions:
            variant, metric_precision = precision_variant(candidate, precision, calibration_loader, calibration_batches)
            metrics: PerfMetrics = collect_metrics(
                variant,
                val_loader,
                device,
                power_watts=power_watts,
                precision=metric_precision,
                latency_multiplier=latency_multiplier,
            )
            violations = memory_violations(metrics.memory_mb, memory_budgets_mb)
            rejected = metrics.memory_mb > active_memory_budget_mb
            row = {
                "pruning_level": pruning,
                "precision": precision,
                "accepted": not rejected,
                "active_budget_mb": active_memory_budget_mb,
                **asdict(metrics),
                **violations,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def pareto_frontier(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    ranked = df[df["accepted"]].sort_values([x_col, "accuracy"], ascending=[True, False]).reset_index(drop=True)
    frontier = []
    best_accuracy = -1.0
    for _, row in ranked.iterrows():
        if row["accuracy"] > best_accuracy:
            frontier.append(row)
            best_accuracy = row["accuracy"]
    return pd.DataFrame(frontier)


def save_plots(df: pd.DataFrame, latency_frontier: pd.DataFrame, energy_frontier: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = df[df["accepted"]]
    rejected = df[~df["accepted"]]

    plt.figure(figsize=(7, 5))
    plt.scatter(accepted["latency_ms"], accepted["accuracy"], c="tab:blue", alpha=0.8, label="Accepted")
    if not rejected.empty:
        plt.scatter(rejected["latency_ms"], rejected["accuracy"], c="tab:gray", alpha=0.5, marker="x", label="Rejected")
    plt.plot(latency_frontier["latency_ms"], latency_frontier["accuracy"], color="red", linewidth=2, label="Pareto")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_latency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(accepted["energy_proxy_j"], accepted["accuracy"], c="tab:green", alpha=0.8, label="Accepted")
    if not rejected.empty:
        plt.scatter(rejected["energy_proxy_j"], rejected["accuracy"], c="tab:gray", alpha=0.5, marker="x", label="Rejected")
    plt.plot(energy_frontier["energy_proxy_j"], energy_frontier["accuracy"], color="red", linewidth=2, label="Pareto")
    plt.xlabel("Energy Proxy (J)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_energy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(accepted["memory_mb"], accepted["accuracy"], c="tab:purple", alpha=0.8, label="Accepted")
    if not rejected.empty:
        plt.scatter(rejected["memory_mb"], rejected["accuracy"], c="tab:gray", alpha=0.5, marker="x", label="Rejected")
    plt.xlabel("Model Memory (MB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Memory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_memory.png", dpi=180)
    plt.close()
