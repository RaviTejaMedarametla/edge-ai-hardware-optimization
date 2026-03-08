import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_opt.hardware import estimate_layerwise_stats
from edge_opt.metrics import collect_metrics
from edge_opt.model import SmallCNN


def _loader() -> DataLoader:
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_collect_metrics_reports_ci_and_energy_note() -> None:
    model = SmallCNN()
    loader = _loader()
    metrics = collect_metrics(
        model,
        loader,
        device=torch.device("cpu"),
        power_watts=2.0,
        precision="fp32",
        benchmark_repeats=1,
        benchmark_trials=2,
    )
    assert metrics.accuracy_ci95_low <= metrics.accuracy <= metrics.accuracy_ci95_high
    assert "not measured" in metrics.energy_proxy_note


def test_estimate_layerwise_stats_uses_dtype_sizes() -> None:
    model = SmallCNN().half()
    df = estimate_layerwise_stats(model, batch_size=2, activation_bytes_per_value=2)
    assert int(df.loc[df["layer"] == "conv1", "parameter_bytes"].iloc[0]) % 2 == 0
