from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _design_space_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"precision": "fp32", "pruning_level": 0.0, "memory_budget_mb": 4.0},
            {"precision": "fp16", "pruning_level": 0.5, "memory_budget_mb": 2.0},
            {"precision": "int8", "pruning_level": 0.7, "memory_budget_mb": 1.0},
        ]
    )


def _research_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study": "baseline_edge_profile",
                "objective": "accuracy_latency_tradeoff",
                "primary_metric": "accuracy",
            },
            {
                "study": "memory_constrained_profile",
                "objective": "acceptance_ratio_under_budget",
                "primary_metric": "accepted_ratio",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hardware table artifacts")
    parser.add_argument("--output-dir", default="outputs/hardware_tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _design_space_table().to_csv(output_dir / "design_space.csv", index=False)
    _research_table().to_csv(output_dir / "research_tables.csv", index=False)


if __name__ == "__main__":
    main()
