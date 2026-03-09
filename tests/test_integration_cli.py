from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_pipeline import main


def test_pipeline_cli_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
seed: 1
dataset: fashion-mnist
batch_size: 4
epochs: 1
learning_rate: 0.001
train_subset: 8
val_subset: 8
power_watts: 2.0
pruning_levels: [0.0]
precisions: [fp32]
calibration_batches: 1
output_dir: {tmp_path / 'out'}
memory_budgets_mb: [10.0]
active_memory_budget_mb: 10.0
cpu_frequency_scale: 1.0
benchmark_repeats: 1
benchmark_trials: 1
benchmark_warmup: 0
num_workers: 0
device: cpu
quantization_backend: fbgemm
fine_tune_epochs: 0
pareto_use_ci: false
""".strip(),
        encoding="utf-8",
    )

    def _fake_build_loaders(*args, **kwargs):
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        return loader, loader

    monkeypatch.setattr("scripts.run_pipeline.build_loaders", _fake_build_loaders)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(cfg)])

    main()

    out_dir = tmp_path / "out"
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "sweep_results.csv").exists()
    assert (out_dir / "reproducibility.json").exists()
