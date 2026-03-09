from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import yaml


def _load_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = ["best_accuracy_accepted", "lowest_latency_ms_accepted"]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        out[key] = {
            "mean": mean,
            "std": var**0.5,
            "min": min(values),
            "max": max(values),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline across multiple seeds")
    parser.add_argument("--config-template", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--output-dir", default="outputs/multi_seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = yaml.safe_load(Path(args.config_template).read_text(encoding="utf-8"))
    all_rows: list[dict[str, float]] = []

    for seed in args.seeds:
        run_config = dict(template)
        run_config["seed"] = seed
        run_config["dataloader_seed"] = seed
        run_output_dir = output_dir / f"seed_{seed}"
        run_config["output_dir"] = str(run_output_dir)

        config_path = output_dir / f"config_seed_{seed}.yaml"
        config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")

        subprocess.check_call(["python", "scripts/run_pipeline.py", "--config", str(config_path)])
        summary = _load_summary(run_output_dir)
        all_rows.append({"seed": seed, **summary})

    aggregates = _aggregate(all_rows)
    (output_dir / "summary_multi_seed.json").write_text(json.dumps({"runs": all_rows, "aggregates": aggregates}, indent=2), encoding="utf-8")

    with open(output_dir / "summary_multi_seed.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "best_accuracy_accepted", "lowest_latency_ms_accepted"])
        for row in all_rows:
            writer.writerow([row["seed"], row.get("best_accuracy_accepted"), row.get("lowest_latency_ms_accepted")])


if __name__ == "__main__":
    main()
