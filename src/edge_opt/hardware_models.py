from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class HardwareModel(ABC):
    """Abstract base class for hardware simulators."""

    @abstractmethod
    def simulate_layer(
        self,
        layer: nn.Module,
        input_shape: tuple[int, ...],
        dtype: torch.dtype,
        batch_size: int,
    ) -> dict[str, float]:
        """
        Return hardware metrics for a single layer.
        Should include at least 'latency_us', 'energy_uj', 'peak_memory_bytes'.
        """

    @abstractmethod
    def simulate_model(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        batch_size: int,
    ) -> dict[str, Any]:
        """
        Return aggregated metrics for the whole model.
        Should include total latency, total energy, peak memory, and per-layer breakdown.
        """


class AnalyticalHardwareModel(HardwareModel):
    """
    Reference analytical model based on roofline and simple MAC/byte accounting.
    This is the default model used in the pipeline.
    """

    def __init__(
        self,
        memory_bandwidth_gbps: float,
        peak_compute_gmacs: float | None = None,
        energy_per_mac_pj: float = 5.0,
        static_power_mw: float = 100.0,
    ):
        self.memory_bandwidth = memory_bandwidth_gbps * 1e9
        self.peak_compute = peak_compute_gmacs * 1e9 if peak_compute_gmacs else float("inf")
        self.energy_per_mac = energy_per_mac_pj * 1e-12
        self.static_power = static_power_mw * 1e-3

    def simulate_layer(
        self,
        layer: nn.Module,
        input_shape: tuple[int, ...],
        dtype: torch.dtype,
        batch_size: int,
    ) -> dict[str, float]:
        return {
            "latency_us": 0.0,
            "energy_uj": 0.0,
            "peak_memory_bytes": 0.0,
        }

    def simulate_model(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        batch_size: int,
    ) -> dict[str, Any]:
        from edge_opt.hardware import estimate_layerwise_stats

        _ = estimate_layerwise_stats(model, batch_size, input_shape)
        raise NotImplementedError("AnalyticalHardwareModel requires a latency measurement.")
