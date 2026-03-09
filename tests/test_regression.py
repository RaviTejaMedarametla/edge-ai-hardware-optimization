from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_opt.experiments import run_sweep
from edge_opt.model import SmallCNN, set_deterministic


def test_minimal_regression_snapshot(tmp_path: Path) -> None:
    set_deterministic(123)
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    df = run_sweep(
        base_model=SmallCNN(),
        train_loader=loader,
        val_loader=loader,
        calibration_loader=loader,
        device=torch.device("cpu"),
        pruning_levels=[0.0],
        precisions=["fp32"],
        power_watts=2.0,
        calibration_batches=1,
        memory_budgets_mb=[10.0],
        active_memory_budget_mb=10.0,
        latency_multiplier=1.0,
        benchmark_repeats=1,
        benchmark_trials=1,
        output_dir=tmp_path,
    )

    current = {
        "accuracy": float(df.iloc[0]["accuracy"]),
        "model_memory_mb": float(df.iloc[0]["model_memory_mb"]),
    }
    baseline_path = Path("tests/baseline_metrics.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert abs(current["accuracy"] - baseline["accuracy"]) < 0.05
    assert abs(current["model_memory_mb"] - baseline["model_memory_mb"]) < 0.05
