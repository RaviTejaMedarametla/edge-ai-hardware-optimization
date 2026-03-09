from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_opt.config import ExperimentConfig
from edge_opt.deploy import deployment_simulation
from edge_opt.model import SmallCNN, resolve_device
from edge_opt.quantization import to_int8


def test_deployment_simulation_rejects_empty_loader() -> None:
    x = torch.randn(0, 1, 28, 28)
    y = torch.randint(0, 10, (0,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    with pytest.raises(ValueError, match="No batches"):
        deployment_simulation(SmallCNN(), loader, device=torch.device("cpu"), cpu_frequency_scale=1.0)


def test_invalid_device_string_warns_and_falls_back() -> None:
    with pytest.warns(UserWarning):
        device = resolve_device("invalid_device")
    assert str(device) == "cpu"


def test_quantization_backend_failure_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    def _raise(*args, **kwargs):
        raise RuntimeError("backend boom")

    monkeypatch.setattr("edge_opt.quantization.get_default_qconfig_mapping", _raise)

    with pytest.warns(UserWarning):
        quantized = to_int8(SmallCNN(), loader, calibration_batches=1, backend="fbgemm", metadata_path=tmp_path / "q.json")
    assert isinstance(quantized, SmallCNN)
    assert (tmp_path / "q.json").exists()


def test_invalid_config_values_raise() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(
            seed=1,
            dataset="fashion-mnist",
            batch_size=-1,
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
            benchmark_trials=1,
        )
