from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from edge_opt.pruning import structured_channel_prune
from edge_opt.model import SmallCNN


@dataclass
class SearchConstraints:
    active_memory_budget_mb: float
    max_energy_j: float


def scale_architecture(model: SmallCNN, width_multiplier: float) -> SmallCNN:
    if not 0.0 < width_multiplier <= 1.0:
        msg = "width_multiplier must be in (0, 1]."
        raise ValueError(msg)

    pruning_level = 1.0 - width_multiplier
    return structured_channel_prune(model, pruning_level=pruning_level)


def select_feasible_models(df: pd.DataFrame, constraints: SearchConstraints) -> pd.DataFrame:
    feasible = df[(df["memory_mb"] <= constraints.active_memory_budget_mb) & (df["energy_proxy_j"] <= constraints.max_energy_j)].copy()
    feasible["feasible"] = True
    return feasible


def choose_best_model(feasible_df: pd.DataFrame) -> dict | None:
    if feasible_df.empty:
        return None

    ranked = feasible_df.sort_values(["accuracy", "latency_ms"], ascending=[False, True]).reset_index(drop=True)
    return ranked.iloc[0].to_dict()
