from pathlib import Path

import pytest

from edge_opt.config import load_config


def test_load_config_with_yaml_parser_and_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 1
dataset: fashion-mnist
batch_size: 32
epochs: 1
learning_rate: 0.001
train_subset: 100
val_subset: 50
power_watts: 3.0
pruning_levels: [0.0, 0.5]
precisions: [fp32, fp16]
calibration_batches: 2
output_dir: outputs
memory_budgets_mb: [1.0]
active_memory_budget_mb: 1.0
cpu_frequency_scale: 0.8
""".strip()
    )

    cfg = load_config(config_path)
    assert cfg.dataloader_seed == 1
    assert cfg.benchmark_trials == 3


def test_load_config_rejects_invalid_pruning(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 1
dataset: fashion-mnist
batch_size: 32
epochs: 1
learning_rate: 0.001
power_watts: 3.0
pruning_levels: [1.2]
precisions: [fp32]
calibration_batches: 2
output_dir: outputs
memory_budgets_mb: [1.0]
active_memory_budget_mb: 1.0
cpu_frequency_scale: 0.8
""".strip()
    )

    with pytest.raises(ValueError):
        load_config(config_path)
