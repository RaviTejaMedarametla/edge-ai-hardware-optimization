from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    seed: int
    dataset: str
    batch_size: int
    epochs: int
    learning_rate: float
    train_subset: int | None
    val_subset: int | None
    power_watts: float
    pruning_levels: list[float]
    precisions: list[str]
    calibration_batches: int
    output_dir: str
    memory_budgets_mb: list[float]
    active_memory_budget_mb: float
    cpu_frequency_scale: float
    dataloader_seed: int
    num_workers: int
    benchmark_repeats: int
    memory_bandwidth_gbps: float
    benchmark_trials: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("batch_size, epochs, and learning_rate must be > 0")
        if any((level < 0.0 or level >= 1.0) for level in self.pruning_levels):
            raise ValueError("all pruning levels must be in [0.0, 1.0)")
        if not self.pruning_levels:
            raise ValueError("pruning_levels must not be empty")
        if not self.memory_budgets_mb or any(value <= 0 for value in self.memory_budgets_mb):
            raise ValueError("memory_budgets_mb must contain positive values")
        allowed_precisions = {"fp32", "fp16", "int8"}
        if not self.precisions or any(p not in allowed_precisions for p in self.precisions):
            raise ValueError("precisions must be non-empty and within {'fp32','fp16','int8'}")


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required config key '{key}'")
    return raw[key]


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return ExperimentConfig(
        seed=int(_require(raw, "seed")),
        dataset=str(_require(raw, "dataset")),
        batch_size=int(_require(raw, "batch_size")),
        epochs=int(_require(raw, "epochs")),
        learning_rate=float(_require(raw, "learning_rate")),
        train_subset=int(raw["train_subset"]) if raw.get("train_subset") is not None else None,
        val_subset=int(raw["val_subset"]) if raw.get("val_subset") is not None else None,
        power_watts=float(_require(raw, "power_watts")),
        pruning_levels=[float(v) for v in _require(raw, "pruning_levels")],
        precisions=[str(v) for v in _require(raw, "precisions")],
        calibration_batches=int(_require(raw, "calibration_batches")),
        output_dir=str(_require(raw, "output_dir")),
        memory_budgets_mb=[float(v) for v in _require(raw, "memory_budgets_mb")],
        active_memory_budget_mb=float(_require(raw, "active_memory_budget_mb")),
        cpu_frequency_scale=float(_require(raw, "cpu_frequency_scale")),
        dataloader_seed=int(raw.get("dataloader_seed", _require(raw, "seed"))),
        num_workers=int(raw.get("num_workers", 2)),
        benchmark_repeats=int(raw.get("benchmark_repeats", 5)),
        memory_bandwidth_gbps=float(raw.get("memory_bandwidth_gbps", 12.8)),
        benchmark_trials=int(raw.get("benchmark_trials", 3)),
    )
