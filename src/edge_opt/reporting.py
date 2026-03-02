from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_summary(
    baseline: dict[str, Any],
    memory_budgets_mb: list[float],
    active_memory_budget_mb: float,
    cpu_frequency_scale: float,
    latency_multiplier: float,
    sweep_df: pd.DataFrame,
    deployment: dict[str, float],
) -> dict[str, Any]:
    accepted = sweep_df[sweep_df["accepted"]]
    return {
        "baseline": baseline,
        "memory_budgets_mb": memory_budgets_mb,
        "active_memory_budget_mb": active_memory_budget_mb,
        "cpu_frequency_scale": cpu_frequency_scale,
        "latency_multiplier": latency_multiplier,
        "study_rows": len(sweep_df),
        "accepted_rows": int(sweep_df["accepted"].sum()),
        "rejected_rows": int((~sweep_df["accepted"]).sum()),
        "best_accuracy_accepted": float(accepted["accuracy"].max()) if not accepted.empty else None,
        "lowest_latency_ms_accepted": float(accepted["latency_ms"].min()) if not accepted.empty else None,
        "deployment": deployment,
    }


def write_outputs(
    output_dir: Path,
    sweep_df: pd.DataFrame,
    latency_frontier: pd.DataFrame,
    energy_frontier: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(output_dir / "sweep_results.csv", index=False)
    latency_frontier.to_csv(output_dir / "pareto_frontier_latency.csv", index=False)
    energy_frontier.to_csv(output_dir / "pareto_frontier_energy.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
