from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_opt.config import load_config
from edge_opt.data import build_loaders
from edge_opt.deploy import deployment_simulation
from edge_opt.experiments import pareto_frontier, run_sweep, save_plots, train_model
from edge_opt.metrics import collect_metrics, memory_violations
from edge_opt.model import SmallCNN, set_deterministic

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-aware edge AI optimization pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic(cfg.seed)
    device = torch.device("cpu")
    latency_multiplier = 1.0 / max(cfg.cpu_frequency_scale, 1e-6)

    train_loader, val_loader = build_loaders(
        dataset_name=cfg.dataset,
        batch_size=cfg.batch_size,
        train_subset=cfg.train_subset,
        val_subset=cfg.val_subset,
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
    )
    baseline_violations = memory_violations(baseline_metrics.memory_mb, cfg.memory_budgets_mb)

    sweep_df = run_sweep(
        base_model=baseline_model,
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
    )

    latency_frontier = pareto_frontier(sweep_df, x_col="latency_ms")
    energy_frontier = pareto_frontier(sweep_df, x_col="energy_proxy_j")
    save_plots(sweep_df, latency_frontier, energy_frontier, output_dir)

    deploy_stats = deployment_simulation(baseline_model, val_loader, cpu_frequency_scale=cfg.cpu_frequency_scale)

    summary = {
        "baseline": {
            **baseline_metrics.__dict__,
            **baseline_violations,
            "accepted_under_active_budget": baseline_metrics.memory_mb <= cfg.active_memory_budget_mb,
        },
        "memory_budgets_mb": cfg.memory_budgets_mb,
        "active_memory_budget_mb": cfg.active_memory_budget_mb,
        "cpu_frequency_scale": cfg.cpu_frequency_scale,
        "latency_multiplier": latency_multiplier,
        "study_rows": len(sweep_df),
        "accepted_rows": int(sweep_df["accepted"].sum()),
        "rejected_rows": int((~sweep_df["accepted"]).sum()),
        "best_accuracy_accepted": float(sweep_df[sweep_df["accepted"]]["accuracy"].max()) if sweep_df["accepted"].any() else None,
        "lowest_latency_ms_accepted": float(sweep_df[sweep_df["accepted"]]["latency_ms"].min()) if sweep_df["accepted"].any() else None,
        "deployment": deploy_stats,
    }

    sweep_df.to_csv(output_dir / "sweep_results.csv", index=False)
    latency_frontier.to_csv(output_dir / "pareto_frontier_latency.csv", index=False)
    energy_frontier.to_csv(output_dir / "pareto_frontier_energy.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
