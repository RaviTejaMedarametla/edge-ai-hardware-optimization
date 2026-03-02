from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return [_parse_scalar(item) for item in items]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, value = stripped.split(":", maxsplit=1)
            raw[key.strip()] = _parse_scalar(value)

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
        dataloader_seed=int(raw.get("dataloader_seed", raw["seed"])),
        num_workers=int(raw.get("num_workers", 2)),
        benchmark_repeats=int(raw.get("benchmark_repeats", 5)),
        memory_bandwidth_gbps=float(raw.get("memory_bandwidth_gbps", 12.8)),
    )
