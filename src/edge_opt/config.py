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


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    return ExperimentConfig(
        seed=raw["seed"],
        dataset=raw["dataset"],
        batch_size=raw["batch_size"],
        epochs=raw["epochs"],
        learning_rate=raw["learning_rate"],
        train_subset=raw.get("train_subset"),
        val_subset=raw.get("val_subset"),
        power_watts=raw["power_watts"],
        pruning_levels=list(raw["pruning_levels"]),
        precisions=list(raw["precisions"]),
        calibration_batches=raw["calibration_batches"],
        output_dir=raw["output_dir"],
        memory_budgets_mb=list(raw["memory_budgets_mb"]),
        active_memory_budget_mb=float(raw["active_memory_budget_mb"]),
        cpu_frequency_scale=float(raw["cpu_frequency_scale"]),
    )
