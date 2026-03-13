from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch

from edge_opt.config import load_config
from edge_opt.data import build_loaders
from edge_opt.deploy import deployment_simulation
from edge_opt.experiments import pareto_frontier, run_sweep, save_plots, train_model
from edge_opt.hardware import estimate_layerwise_stats, precision_tradeoff_table, save_hardware_artifacts, summarize_hardware
from edge_opt.metrics import collect_metrics, memory_violations
from edge_opt.model import SmallCNN, resolve_device, set_deterministic
from edge_opt.reporting import build_summary, write_outputs


def _pip_freeze() -> list[str]:
    try:
        output = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-aware edge AI optimization pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic(cfg.seed)
    device = resolve_device(cfg.device)
    latency_multiplier = 1.0 / max(cfg.cpu_frequency_scale, 1e-6)

    train_loader, val_loader = build_loaders(
        dataset_name=cfg.dataset,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        val_subset=cfg.val_subset,
        seed=cfg.dataloader_seed,
        num_workers=cfg.num_workers,
    )

    baseline_model = SmallCNN()
    baseline_model = train_model(
        baseline_model,
        train_loader,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        device=device,
    )
    baseline_model.eval()

    baseline_metrics = collect_metrics(
        baseline_model,
        val_loader,
        device=device,
        power_watts=cfg.power_watts,
        precision="fp32",
        latency_multiplier=latency_multiplier,
        benchmark_repeats=cfg.benchmark_repeats,
        benchmark_trials=cfg.benchmark_trials,
        benchmark_warmup=cfg.benchmark_warmup,
    )
    baseline_violations = memory_violations(baseline_metrics.model_memory_mb, cfg.memory_budgets_mb)

    sweep_df = run_sweep(
        base_model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        calibration_loader=train_loader,
        device=device,
        pruning_levels=cfg.pruning_levels,
        precisions=cfg.precisions,
        power_watts=cfg.power_watts,
        calibration_batches=cfg.calibration_batches,
        memory_budgets_mb=cfg.memory_budgets_mb,
        active_memory_budget_mb=cfg.active_memory_budget_mb,
        latency_multiplier=latency_multiplier,
        benchmark_repeats=cfg.benchmark_repeats,
        benchmark_trials=cfg.benchmark_trials,
        benchmark_warmup=cfg.benchmark_warmup,
        fine_tune_epochs=cfg.fine_tune_epochs,
        learning_rate=cfg.learning_rate,
        quantization_backend=cfg.quantization_backend,
        output_dir=output_dir,
    )

    latency_frontier = pareto_frontier(sweep_df, x_col="latency_ms", use_ci=cfg.pareto_use_ci)
    energy_frontier = pareto_frontier(sweep_df, x_col="energy_proxy_j", use_ci=cfg.pareto_use_ci)
    save_plots(sweep_df, latency_frontier, energy_frontier, output_dir, show_error_bars=cfg.pareto_use_ci)

    layerwise_df = estimate_layerwise_stats(baseline_model, batch_size=cfg.batch_size)
    hardware_summary = summarize_hardware(
        layerwise_df,
        latency_ms=baseline_metrics.latency_ms,
        memory_bandwidth_gbps=cfg.memory_bandwidth_gbps,
        peak_compute_gmacs=cfg.peak_compute_gmacs,
    )
    precision_df = precision_tradeoff_table(sweep_df)
    save_hardware_artifacts(output_dir, layerwise_df, precision_df, hardware_summary)

    deploy_stats = deployment_simulation(baseline_model, val_loader, device=device, cpu_frequency_scale=cfg.cpu_frequency_scale)

    baseline_summary = {
        **baseline_metrics.__dict__,
        **baseline_violations,
        "accepted_under_active_budget": baseline_metrics.model_memory_mb <= cfg.active_memory_budget_mb,
    }
    summary = build_summary(
        baseline=baseline_summary,
        memory_budgets_mb=cfg.memory_budgets_mb,
        active_memory_budget_mb=cfg.active_memory_budget_mb,
        cpu_frequency_scale=cfg.cpu_frequency_scale,
        latency_multiplier=latency_multiplier,
        sweep_df=sweep_df,
        deployment={**deploy_stats, **hardware_summary},
    )

    reproducibility = {
        "packages": _pip_freeze(),
        "cpu": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "command": " ".join(["python", "scripts/run_pipeline.py", "--config", args.config]),
        "git_commit_hash": _git_commit_hash(),
        "seed": cfg.seed,
        "dataloader_seed": cfg.dataloader_seed,
        "device_requested": cfg.device,
        "device_resolved": str(device),
    }

    write_outputs(output_dir, sweep_df, latency_frontier, energy_frontier, summary)
    (output_dir / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
