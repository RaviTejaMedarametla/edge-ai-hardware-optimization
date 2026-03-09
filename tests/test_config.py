from pathlib import Path

import pytest

from edge_opt.config import ExperimentConfig, load_config


BASE = """
seed: 1
dataset: fashion-mnist
batch_size: 32
epochs: 1
learning_rate: 0.001
power_watts: 3.0
pruning_levels: [0.0, 0.5]
precisions: [fp32, fp16]
calibration_batches: 2
output_dir: outputs
memory_budgets_mb: [1.0]
active_memory_budget_mb: 1.0
cpu_frequency_scale: 0.8
""".strip()


def test_load_config_with_yaml_parser_and_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(BASE)

    cfg = load_config(config_path)
    assert cfg.dataloader_seed == 1
    assert cfg.benchmark_trials == 3
    assert cfg.device == "cpu"


def test_load_config_rejects_invalid_pruning(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(BASE.replace("[0.0, 0.5]", "[1.2]"))

    with pytest.raises(ValueError):
        load_config(config_path)


def test_config_rejects_zero_trials() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(
            seed=1,
            dataset="fashion-mnist",
            batch_size=32,
            epochs=1,
            learning_rate=0.001,
            train_subset=None,
            val_subset=None,
            power_watts=2.0,
            pruning_levels=[0.0],
            precisions=["fp32"],
            calibration_batches=1,
            output_dir="outputs",
            memory_budgets_mb=[1.0],
            active_memory_budget_mb=1.0,
            cpu_frequency_scale=1.0,
            dataloader_seed=1,
            num_workers=0,
            benchmark_repeats=1,
            memory_bandwidth_gbps=10.0,
            benchmark_trials=0,
        )
