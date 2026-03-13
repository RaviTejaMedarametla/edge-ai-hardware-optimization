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
from edge_opt.pruning import prune_and_finetune, structured_channel_prune
from edge_opt.quantization import to_fp16, to_int8


def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> nn.Module:
    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    return model


def train_model(model: nn.Module, train_loader: DataLoader, epochs: int, learning_rate: float, device: torch.device) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
    return model


def run_sweep(
    base_model: nn.Module,
    train_loader: DataLoader,
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
    benchmark_repeats: int = 5,
    benchmark_trials: int = 3,
    benchmark_warmup: int = 3,
    fine_tune_epochs: int = 0,
    learning_rate: float = 1e-3,
    quantization_backend: str = "fbgemm",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    error_rows: list[dict] = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for pruning in pruning_levels:
        if fine_tune_epochs > 0:

            def _epoch(m: nn.Module, loader: DataLoader) -> nn.Module:
                opt = torch.optim.Adam(m.parameters(), lr=learning_rate)
                crit = nn.CrossEntropyLoss()
                m.train()
                for inputs, targets in loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    opt.zero_grad()
                    outputs = m(inputs)
                    loss = crit(outputs, targets)
                    loss.backward()
                    opt.step()
                return m

            pruned_base = prune_and_finetune(base_model, pruning, fine_tune_epochs, train_loader, _epoch)
        else:
            pruned_base = structured_channel_prune(base_model, pruning)

        pruned_base = pruned_base.to(device)

        for precision in precisions:
            try:
                variant = deepcopy(pruned_base)

                if precision == "fp32":
                    pass
                elif precision == "fp16":
                    variant = to_fp16(variant)
                elif precision == "int8":
                    metadata_path = None
                    if output_dir is not None:
                        metadata_path = output_dir / f"quantization_metadata_p{pruning}_{precision}.json"
                    variant = to_int8(
                        variant,
                        calibration_loader,
                        calibration_batches=calibration_batches,
                        metadata_path=metadata_path,
                    )
                else:
                    raise ValueError(f"Unsupported precision '{precision}'")

                metrics: PerfMetrics = collect_metrics(
                    variant,
                    val_loader,
                    device,
                    power_watts=power_watts,
                    precision=precision,
                    latency_multiplier=latency_multiplier,
                    benchmark_repeats=benchmark_repeats,
                    benchmark_trials=benchmark_trials,
                    benchmark_warmup=benchmark_warmup,
                )
                violations = memory_violations(metrics.model_memory_mb, memory_budgets_mb)
                rejected = metrics.model_memory_mb > active_memory_budget_mb
                row = {
                    "pruning_level": pruning,
                    "precision": precision,
                    "accepted": not rejected,
                    "error": None,
                    "active_budget_mb": active_memory_budget_mb,
                    **asdict(metrics),
                    **violations,
                }
                rows.append(row)

            except Exception as exc:
                error_row = {
                    "pruning_level": pruning,
                    "precision": precision,
                    "error": str(exc),
                    "active_budget_mb": active_memory_budget_mb,
                }
                error_rows.append(error_row)

    if output_dir is not None and error_rows:
        error_df = pd.DataFrame(error_rows)
        error_df.to_csv(output_dir / "sweep_errors.csv", index=False)

    return pd.DataFrame(rows)


def pareto_frontier(df: pd.DataFrame, x_col: str, use_ci: bool = False) -> pd.DataFrame:
    ranked = df[df["accepted"]].copy()
    acc_col = "accuracy_ci95_low" if use_ci and "accuracy_ci95_low" in ranked.columns else "accuracy"
    ranked = ranked.sort_values([x_col, acc_col], ascending=[True, False]).reset_index(drop=True)
    frontier = []
    best_accuracy = -1.0
    for _, row in ranked.iterrows():
        if row[acc_col] > best_accuracy:
            frontier.append(row)
            best_accuracy = row[acc_col]
    return pd.DataFrame(frontier)


def save_plots(
    df: pd.DataFrame,
    latency_frontier: pd.DataFrame,
    energy_frontier: pd.DataFrame,
    output_dir: Path,
    show_error_bars: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = df[df["accepted"]]
    rejected = df[~df["accepted"]]

    plt.figure(figsize=(7, 5))
    plt.scatter(accepted["latency_ms"], accepted["accuracy"], c="tab:blue", alpha=0.8, label="Accepted")
    if show_error_bars and "accuracy_ci95_low" in accepted.columns:
        yerr_low = accepted["accuracy"] - accepted["accuracy_ci95_low"]
        yerr_high = accepted["accuracy_ci95_high"] - accepted["accuracy"]
        plt.errorbar(accepted["latency_ms"], accepted["accuracy"], yerr=[yerr_low, yerr_high], fmt="none", ecolor="tab:blue", alpha=0.35)
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
    if show_error_bars and "accuracy_ci95_low" in accepted.columns:
        yerr_low = accepted["accuracy"] - accepted["accuracy_ci95_low"]
        yerr_high = accepted["accuracy_ci95_high"] - accepted["accuracy"]
        plt.errorbar(accepted["energy_proxy_j"], accepted["accuracy"], yerr=[yerr_low, yerr_high], fmt="none", ecolor="tab:green", alpha=0.35)
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
    plt.scatter(accepted["model_memory_mb"], accepted["accuracy"], c="tab:purple", alpha=0.8, label="Accepted")
    if not rejected.empty:
        plt.scatter(rejected["model_memory_mb"], rejected["accuracy"], c="tab:gray", alpha=0.5, marker="x", label="Rejected")
    plt.xlabel("Model Memory (MB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Memory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_memory.png", dpi=180)
    plt.close()
