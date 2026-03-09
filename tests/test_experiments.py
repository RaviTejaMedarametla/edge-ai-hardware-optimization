from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_opt.experiments import pareto_frontier, run_sweep
from edge_opt.model import SmallCNN


def _dummy_loader() -> DataLoader:
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_pareto_frontier_filters_dominated_points() -> None:
    df = pd.DataFrame(
        [
            {"accepted": True, "latency_ms": 10.0, "accuracy": 0.80, "accuracy_ci95_low": 0.78},
            {"accepted": True, "latency_ms": 8.0, "accuracy": 0.78, "accuracy_ci95_low": 0.76},
            {"accepted": True, "latency_ms": 12.0, "accuracy": 0.79, "accuracy_ci95_low": 0.70},
        ]
    )
    frontier = pareto_frontier(df, "latency_ms", use_ci=True)
    assert list(frontier["latency_ms"]) == [8.0, 10.0]


def test_run_sweep_captures_variant_errors() -> None:
    loader = _dummy_loader()
    model = SmallCNN()
    out = run_sweep(
        base_model=model,
        train_loader=loader,
        val_loader=loader,
        calibration_loader=loader,
        device=torch.device("cpu"),
        pruning_levels=[0.0],
        precisions=["fp32", "bad_precision"],
        power_watts=2.0,
        calibration_batches=1,
        memory_budgets_mb=[1.0],
        active_memory_budget_mb=2.0,
        latency_multiplier=1.0,
        benchmark_repeats=1,
        benchmark_trials=1,
        output_dir=Path("outputs/test"),
    )
    assert len(out) == 2
    assert out["error"].isna().sum() == 1
    assert out["error"].notna().sum() == 1
