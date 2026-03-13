import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edge_opt.hardware import estimate_layerwise_stats, summarize_hardware
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
    assert metrics.estimated_runtime_memory_mb > metrics.model_memory_mb


def test_collect_metrics_rejects_empty_loader() -> None:
    x = torch.randn(0, 1, 28, 28)
    y = torch.randint(0, 10, (0,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    with torch.no_grad():
        model = SmallCNN()
        try:
            collect_metrics(model, loader, torch.device("cpu"), 2.0, "fp32")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "No batches" in str(exc)


def test_estimate_layerwise_stats_uses_dtype_sizes() -> None:
    model = SmallCNN().half()
    df = estimate_layerwise_stats(model, batch_size=2, activation_bytes_per_value=2)
    assert int(df.loc[df["layer"] == "conv1", "parameter_bytes"].iloc[0]) % 2 == 0


def test_estimate_layerwise_stats_supports_generic_cnn() -> None:
    class TinyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.head = nn.Linear(4 * 28 * 28, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            return self.head(x.flatten(start_dim=1))

    df = estimate_layerwise_stats(TinyNet(), batch_size=2)
    assert set(df["layer"].tolist()) == {"stem", "head"}
    assert (df["macs"] > 0).all()


def test_summarize_hardware_reports_roofline_bound() -> None:
    layerwise_df = estimate_layerwise_stats(SmallCNN(), batch_size=2)
    summary = summarize_hardware(layerwise_df, latency_ms=2.0, memory_bandwidth_gbps=10.0, peak_compute_gmacs=1.0)
    assert "bound_regime" in summary
    assert summary["bound_regime"] in {"memory-bound", "compute-bound"}
