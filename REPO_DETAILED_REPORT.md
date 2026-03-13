# Complete Repository Line-by-Line Report

This document inventories every tracked file and reproduces each line with line numbers for full auditability.

## `.github/workflows/ci.yml`

- Total lines: **20**
- Purpose: Repository metadata / project-level artifact.

```text
   1: name: CI
   2: 
   3: on:
   4:   push:
   5:   pull_request:
   6: 
   7: jobs:
   8:   test:
   9:     runs-on: ubuntu-latest
  10:     steps:
  11:       - uses: actions/checkout@v4
  12:       - uses: actions/setup-python@v5
  13:         with:
  14:           python-version: '3.11'
  15:       - name: Install dependencies
  16:         run: |
  17:           python -m pip install --upgrade pip
  18:           pip install -r requirements.txt
  19:       - name: Run tests
  20:         run: PYTHONPATH=src pytest -q
```

## `.github/workflows/commitlint.yml`

- Total lines: **18**
- Purpose: Repository metadata / project-level artifact.

```text
   1: name: Commitlint
   2: 
   3: on:
   4:   pull_request:
   5:     types: [opened, synchronize, reopened, edited]
   6: 
   7: jobs:
   8:   commitlint:
   9:     runs-on: ubuntu-latest
  10:     steps:
  11:       - uses: actions/checkout@v4
  12:         with:
  13:           fetch-depth: 0
  14:       - uses: actions/setup-node@v4
  15:         with:
  16:           node-version: '20'
  17:       - run: npm install --no-save @commitlint/cli @commitlint/config-conventional
  18:       - run: npx commitlint --from=${{ github.event.pull_request.base.sha }} --to=${{ github.event.pull_request.head.sha }}
```

## `.gitignore`

- Total lines: **9**
- Purpose: Repository metadata / project-level artifact.

```text
   1: __pycache__/
   2: *.pyc
   3: .venv/
   4: data/
   5: outputs/
   6: .ipynb_checkpoints/
   7: .idea/
   8: .vscode/
   9: *.log
```

## `LICENSE`

- Total lines: **21**
- Purpose: Repository metadata / project-level artifact.

```text
   1: MIT License
   2: 
   3: Copyright (c) 2026 RaviTejaMedarametla
   4: 
   5: Permission is hereby granted, free of charge, to any person obtaining a copy
   6: of this software and associated documentation files (the "Software"), to deal
   7: in the Software without restriction, including without limitation the rights
   8: to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   9: copies of the Software, and to permit persons to whom the Software is
  10: furnished to do so, subject to the following conditions:
  11: 
  12: The above copyright notice and this permission notice shall be included in all
  13: copies or substantial portions of the Software.
  14: 
  15: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  16: IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  17: FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  18: AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  19: LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  20: OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  21: SOFTWARE.
```

## `README.md`

- Total lines: **150**
- Purpose: Repository metadata / project-level artifact.

```text
   1: # Hardware-Aware Machine Learning Pipeline for Edge Deployment
   2: 
   3: ![CI](https://github.com/RaviTejaMedarametla/edge-ai-hardware-optimization/actions/workflows/ci.yml/badge.svg)
   4: 
   5: A reproducible research-oriented framework for training, compressing, and benchmarking compact neural networks under edge hardware constraints.
   6: 
   7: ## Overview
   8: This repository implements an end-to-end machine learning workflow for studying model behavior in resource-constrained environments. The system couples deterministic training and evaluation with hardware-aware analysis to quantify trade-offs among predictive performance, latency, memory footprint, and energy proxy metrics.
   9: 
  10: The project addresses a central challenge in edge AI engineering: model quality alone is insufficient when deployment targets have strict compute and memory limits. By integrating pruning, precision-aware evaluation, constraint filtering, and structured artifact export in a single configuration-driven pipeline, the repository supports repeatable experiments and transparent comparison of optimization decisions.
  11: 
  12: ## Project Motivation
  13: Modern edge deployments require models that satisfy application-level constraints (e.g., latency and memory budgets) while maintaining acceptable task performance. This repository is motivated by three research and engineering priorities:
  14: 
  15: - **Edge feasibility:** evaluate compact CNN variants for scenarios where compute and memory resources are bounded.
  16: - **Hardware-aware optimization:** measure the impact of structured pruning and numeric precision choices on systems-oriented metrics.
  17: - **Deterministic experimentation:** ensure that repeated runs remain comparable through fixed seeds, controlled data loading, and explicit configuration.
  18: 
  19: ## System Architecture
  20: The pipeline is organized as modular components that mirror a typical ML systems research workflow:
  21: 
  22: - **Data Pipeline**  
  23:   Builds train/validation loaders for configured datasets and subset sizes with deterministic controls.
  24: 
  25: - **Model Training**  
  26:   Trains a baseline compact CNN under fixed optimization settings.
  27: 
  28: - **Model Compression**  
  29:   Applies structured channel pruning sweeps and precision variants (FP32, FP16, INT8).
  30: 
  31: - **Hardware-Aware Evaluation**  
  32:   Computes accuracy and systems metrics, including latency, throughput, memory usage, and energy proxy; enforces memory-budget constraints; and derives Pareto frontiers.
  33: 
  34: - **Inference / Deployment Analysis**  
  35:   Produces deployment-oriented summaries, layer-wise hardware statistics, and tabular/plot artifacts for downstream comparison.
  36: 
  37: ## Repository Structure
  38: - **`src/edge_opt/`**  
  39:   Core implementation of configuration loading, data handling, model definition, training/evaluation orchestration, pruning/quantization, metrics, hardware analysis, deployment simulation, and reporting.
  40: 
  41: - **`scripts/`**  
  42:   CLI entry points for running the full hardware-aware optimization pipeline.
  43: 
  44: - **`configs/`**  
  45:   YAML experiment configurations controlling seeds, dataset subsets, pruning levels, precision modes, and hardware budget settings.
  46: 
  47: - **`docs/`**  
  48:   Supplemental technical notes on architecture and hardware analysis.
  49: 
  50: - **`outputs/`** *(generated at runtime)*  
  51:   Experiment artifacts such as sweep tables, frontier CSV files, summary metadata, and analysis plots.
  52: 
  53: ## Features
  54: - Deterministic, configuration-driven ML experimentation.
  55: - Hardware-aware benchmarking with mean/median/p95 latency, throughput, parameter memory, runtime memory estimate, and energy proxy metrics.
  56: - Structured pruning sweeps with precision-aware comparisons.
  57: - Constraint-first evaluation via configurable memory budgets.
  58: - Automated artifact generation for reproducible analysis.
  59: - CLI-based pipeline execution for consistent experiment orchestration.
  60: - Device-aware execution (`cpu`/`cuda`/`mps`) with safe fallback warnings.
  61: - Optional post-pruning fine-tuning and configurable quantization backends.
  62: - Reproducibility metadata export (`reproducibility.json`) and multi-seed orchestration script.
  63: 
  64: 
  65: ## Installation
  66: ```bash
  67: python -m venv .venv
  68: source .venv/bin/activate
  69: pip install -r requirements.txt
  70: export PYTHONPATH=src
  71: ```
  72: 
  73: ## Usage
  74: Run the default experiment pipeline:
  75: 
  76: ```bash
  77: python scripts/run_pipeline.py --config configs/default.yaml
  78: ```
  79: 
  80: To use a custom configuration, provide a different YAML file:
  81: 
  82: ```bash
  83: python scripts/run_pipeline.py --config <path-to-config>.yaml
  84: ```
  85: 
  86: Run multi-seed orchestration:
  87: 
  88: ```bash
  89: python experiments/multi_seed.py --config-template configs/default.yaml --seeds 1 2 3
  90: ```
  91: 
  92: ## Quick Start
  93: Run a complete default experiment and inspect generated artifacts:
  94: 
  95: ```bash
  96: python scripts/run_pipeline.py --config configs/default.yaml
  97: ```
  98: 
  99: Expected outcomes after a successful run:
 100: - Optional roofline-style regime tagging (`memory-bound` vs `compute-bound`) is produced when `peak_compute_gmacs` is configured.
 101: - Console prints `summary.json`-style metrics (accuracy, latency, memory, energy proxy, and acceptance stats).
 102: - `outputs/` contains analysis artifacts such as:
 103:   - `sweep_results.csv`
 104:   - `pareto_frontier_latency.csv`
 105:   - `pareto_frontier_energy.csv`
 106:   - `accuracy_vs_latency.png`, `accuracy_vs_energy.png`, `accuracy_vs_memory.png`
 107: 
 108: Example summary snippet (illustrative):
 109: 
 110: ```json
 111: {
 112:   "study_rows": 12,
 113:   "accepted_rows": 9,
 114:   "best_accuracy_accepted": 0.88,
 115:   "lowest_latency_ms_accepted": 3.42
 116: }
 117: ```
 118: 
 119: ## Reproducibility
 120: Reproducibility is supported through explicit configuration and deterministic controls:
 121: 
 122: - **Configuration files:** experiment settings are specified in YAML (e.g., dataset subsets, optimization sweeps, and hardware constraints).
 123: - **Deterministic seeds:** global and dataloader seeds are configured to reduce run-to-run variability.
 124: - **Experiment artifacts:** each run writes structured outputs (tables, frontiers, summaries, and plots) to enable traceable comparisons.
 125: 
 126: For strict comparability across runs, keep seed values, dataset subsets, batch size, benchmark repeat counts, and memory-budget settings fixed.
 127: 
 128: ## Related Projects
 129: This repository is part of a broader portfolio focused on hardware-aware machine learning, edge AI optimization, deterministic ML pipelines, and production ML systems.
 130: 
 131: - `neural-network-from-scratch`
 132: - `classification-of-handwritten-digits1`
 133: - `edge-ai-hardware-optimization`
 134: - `data-analysis-for-hospitals`
 135: - `nba-data-preprocessing`
 136: 
 137: ## Development Notes
 138: This repository was recently refactored to improve clarity, reliability, and reproducibility (configuration validation, architecture generalization, stronger testing, and CI automation).
 139: 
 140: The refactor was implementation-focused: core experiment intent and pipeline semantics remain the same, but the codebase now provides clearer invariants, better failure handling, and more transparent benchmarking outputs.
 141: 
 142: ## Future Work
 143: Potential extensions include:
 144: 
 145: - Deployment studies on embedded and heterogeneous edge hardware.
 146: - Additional compression strategies beyond current pruning/precision workflows.
 147: - Expanded benchmarking methodology for improved hardware realism and cross-platform comparability.
 148: 
 149: ## License
 150: This project is released under the terms of the license provided in [`LICENSE`](LICENSE).
```

## `commitlint.config.js`

- Total lines: **3**
- Purpose: Repository metadata / project-level artifact.

```text
   1: module.exports = {
   2:   extends: ['@commitlint/config-conventional'],
   3: };
```

## `configs/default.yaml`

- Total lines: **27**
- Purpose: Experiment configuration template.

```text
   1: seed: 7
   2: dataloader_seed: 7
   3: num_workers: 2
   4: benchmark_repeats: 5
   5: benchmark_trials: 3
   6: benchmark_warmup: 3
   7: memory_bandwidth_gbps: 12.8
   8: device: cpu
   9: quantization_backend: fbgemm
  10: fine_tune_epochs: 0
  11: pareto_use_ci: false
  12: dataset: fashion-mnist
  13: batch_size: 128
  14: epochs: 2
  15: learning_rate: 0.001
  16: train_subset: 12000
  17: val_subset: 3000
  18: power_watts: 5.0
  19: pruning_levels: [0.0, 0.25, 0.5, 0.7]
  20: precisions: [fp32, fp16, int8]
  21: calibration_batches: 8
  22: memory_budgets_mb: [1.0, 2.0, 4.0]
  23: active_memory_budget_mb: 2.0
  24: cpu_frequency_scale: 0.7
  25: output_dir: outputs
  26: 
  27: peak_compute_gmacs: 64.0
```

## `docs/architecture.md`

- Total lines: **36**
- Purpose: Project technical documentation.

```text
   1: # Architecture Notes
   2: 
   3: ## Pipeline stages
   4: 
   5: 1. **Configuration load** (`edge_opt.config`): parse YAML into a typed dataclass.
   6: 2. **Dataset and loader setup** (`edge_opt.data`): build deterministic train/validation loaders.
   7: 3. **Baseline training** (`edge_opt.experiments.train_model`): train compact CNN.
   8: 4. **Optimization sweep** (`edge_opt.experiments.run_sweep`): apply pruning and precision variants.
   9: 5. **Metric collection** (`edge_opt.metrics`): compute accuracy, latency, throughput, memory, and energy proxy.
  10: 6. **Constraint filtering**: classify candidates by active memory budget.
  11: 7. **Reporting**: save sweep tables, Pareto frontiers, plots, and summary JSON.
  12: 
  13: ## Design decisions
  14: 
  15: - A compact CNN is used to keep iteration cycle times short while retaining realistic convolutional operator behavior.
  16: - Structured pruning removes whole channels to preserve dense kernels and straightforward deployment compatibility.
  17: - Precision conversion is explicit (`fp32`, `fp16`, `int8`) to keep evaluation paths auditable.
  18: - Pareto frontier generation is performed after constraint filtering to avoid infeasible configurations.
  19: 
  20: ## Operational constraints
  21: 
  22: - CPU execution only in the default pipeline.
  23: - No distributed training support.
  24: - Quantization backend defaults to `fbgemm`.
  25: 
  26: ## Deployment challenges
  27: 
  28: - Batch-size sensitivity can mask single-request latency behavior.
  29: - Memory headroom margins in production typically require tighter limits than nominal model-size estimates.
  30: - Host-level contention (co-scheduled workloads, thermal throttling) can significantly alter latency distributions.
  31: 
  32: ## Recommended extensions
  33: 
  34: - Add multi-seed experiment orchestration and confidence intervals.
  35: - Integrate hardware counters for cache, bandwidth, and instruction-level profiling.
  36: - Introduce artifact manifests with model checksum and dataset version metadata.
```

## `docs/hardware-analysis.md`

- Total lines: **83**
- Purpose: Project technical documentation.

```text
   1: # Hardware Analysis Methodology
   2: 
   3: ## Objective
   4: 
   5: This repository provides first-order hardware-aware estimates for memory pressure, compute density, precision trade-offs, and roofline-style bottleneck diagnosis without changing the baseline model API.
   6: 
   7: ## Layer-wise breakdown
   8: 
   9: `layerwise_breakdown.csv` reports per-layer statistics for supported trainable modules (`Conv2d`, `Linear`):
  10: 
  11: - output activation elements
  12: - parameter bytes
  13: - activation bytes
  14: - MAC estimates
  15: 
  16: These values are shape-derived estimates from the baseline batch configuration and are intended for relative comparison.
  17: 
  18: ### Generic profiling support
  19: 
  20: Layer-wise profiling no longer assumes a fixed model topology. Instead, forward hooks are attached dynamically to each supported module. This allows profiling of scratch-built CNNs/MLPs that differ from the default `SmallCNN` architecture while preserving deterministic output tables.
  21: 
  22: ### Dtype-aware byte accounting
  23: 
  24: The hardware estimator is dtype-aware:
  25: 
  26: - **Parameter bytes** are inferred from each module's tensor dtype via `tensor.element_size()`.
  27: - **Activation bytes** can be configured through `activation_bytes_per_value` (or default to the same byte width as parameters).
  28: 
  29: This is more realistic than assuming a fixed byte width and better reflects FP32/FP16 deployment studies.
  30: 
  31: ## Bandwidth and utilization estimate
  32: 
  33: `hardware_summary.csv` derives:
  34: 
  35: - estimated total bytes moved (parameters + activations)
  36: - estimated total MACs
  37: - achieved bandwidth approximation (`bytes / latency`)
  38: - configured memory bandwidth (`memory_bandwidth_gbps`)
  39: - bandwidth utilization ratio
  40: - achieved GMAC/s estimate
  41: - arithmetic intensity (`MACs / byte`)
  42: 
  43: ## Roofline-oriented bound tagging
  44: 
  45: When `peak_compute_gmacs` is configured in the experiment YAML, hardware summary adds:
  46: 
  47: - configured peak compute throughput
  48: - roofline knee (`peak_compute_gmacs / memory_bandwidth_gbps`)
  49: - inferred bottleneck regime: `memory-bound` or `compute-bound`
  50: 
  51: This provides a practical, research-friendly interpretation of whether optimization should prioritize reduced memory traffic (e.g., pruning/layout) or improved arithmetic throughput (e.g., lower precision/hardware kernels).
  52: 
  53: ## Precision and quantization trade-off table
  54: 
  55: `precision_tradeoffs.csv` aggregates sweep results by precision mode:
  56: 
  57: - mean accuracy
  58: - mean latency
  59: - mean memory footprint
  60: - mean energy proxy
  61: - acceptance ratio under active memory budget
  62: 
  63: ## Energy proxy interpretation
  64: 
  65: Energy is reported as a **proxy**, not direct hardware power telemetry:
  66: 
  67: - `energy_proxy_j = power_watts × latency_seconds`
  68: - `power_watts` is a static configuration parameter
  69: - this value should be interpreted as a comparative indicator across variants, not measured joules from on-device instrumentation
  70: 
  71: ## Failure modes and caveats
  72: 
  73: - Estimates do not include cache-miss penalties or kernel launch overhead details.
  74: - INT8 execution path may vary by backend implementation and calibration data quality.
  75: - CPU host contention can significantly affect measured latency and derived utilization.
  76: - Activation memory reported is per-layer output footprint and not full runtime peak memory.
  77: - Energy proxy is not a substitute for measured board-level power traces.
  78: - Roofline labeling is first-order and depends on accurate `peak_compute_gmacs` and memory-bandwidth configuration.
  79: 
  80: ## Edge and constrained scenarios
  81: 
  82: For low-memory systems, reduce `active_memory_budget_mb` and compare acceptance ratio changes in `precision_tradeoffs.csv`.
  83: For bandwidth-constrained studies, lower `memory_bandwidth_gbps` to stress utilization estimates and identify compute- vs transfer-bound regions.
```

## `experiments/multi_seed.py`

- Total lines: **72**
- Purpose: Executable orchestration script.

```text
   1: from __future__ import annotations
   2: 
   3: import argparse
   4: import csv
   5: import json
   6: import subprocess
   7: from pathlib import Path
   8: 
   9: import yaml
  10: 
  11: 
  12: def _load_summary(path: Path) -> dict:
  13:     return json.loads((path / "summary.json").read_text(encoding="utf-8"))
  14: 
  15: 
  16: def _aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
  17:     keys = ["best_accuracy_accepted", "lowest_latency_ms_accepted"]
  18:     out: dict[str, dict[str, float]] = {}
  19:     for key in keys:
  20:         values = [float(r[key]) for r in rows if r.get(key) is not None]
  21:         if not values:
  22:             continue
  23:         mean = sum(values) / len(values)
  24:         var = sum((v - mean) ** 2 for v in values) / len(values)
  25:         out[key] = {
  26:             "mean": mean,
  27:             "std": var**0.5,
  28:             "min": min(values),
  29:             "max": max(values),
  30:         }
  31:     return out
  32: 
  33: 
  34: def main() -> None:
  35:     parser = argparse.ArgumentParser(description="Run pipeline across multiple seeds")
  36:     parser.add_argument("--config-template", required=True)
  37:     parser.add_argument("--seeds", nargs="+", required=True, type=int)
  38:     parser.add_argument("--output-dir", default="outputs/multi_seed")
  39:     args = parser.parse_args()
  40: 
  41:     output_dir = Path(args.output_dir)
  42:     output_dir.mkdir(parents=True, exist_ok=True)
  43: 
  44:     template = yaml.safe_load(Path(args.config_template).read_text(encoding="utf-8"))
  45:     all_rows: list[dict[str, float]] = []
  46: 
  47:     for seed in args.seeds:
  48:         run_config = dict(template)
  49:         run_config["seed"] = seed
  50:         run_config["dataloader_seed"] = seed
  51:         run_output_dir = output_dir / f"seed_{seed}"
  52:         run_config["output_dir"] = str(run_output_dir)
  53: 
  54:         config_path = output_dir / f"config_seed_{seed}.yaml"
  55:         config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
  56: 
  57:         subprocess.check_call(["python", "scripts/run_pipeline.py", "--config", str(config_path)])
  58:         summary = _load_summary(run_output_dir)
  59:         all_rows.append({"seed": seed, **summary})
  60: 
  61:     aggregates = _aggregate(all_rows)
  62:     (output_dir / "summary_multi_seed.json").write_text(json.dumps({"runs": all_rows, "aggregates": aggregates}, indent=2), encoding="utf-8")
  63: 
  64:     with open(output_dir / "summary_multi_seed.csv", "w", newline="", encoding="utf-8") as file:
  65:         writer = csv.writer(file)
  66:         writer.writerow(["seed", "best_accuracy_accepted", "lowest_latency_ms_accepted"])
  67:         for row in all_rows:
  68:             writer.writerow([row["seed"], row.get("best_accuracy_accepted"), row.get("lowest_latency_ms_accepted")])
  69: 
  70: 
  71: if __name__ == "__main__":
  72:     main()
```

## `requirements.txt`

- Total lines: **6**
- Purpose: Repository metadata / project-level artifact.

```text
   1: torch
   2: torchvision
   3: matplotlib
   4: pandas
   5: pyyaml
   6: pytest
```

## `scripts/run_pipeline.py`

- Total lines: **156**
- Purpose: Executable orchestration script.

```text
   1: from __future__ import annotations
   2: 
   3: import argparse
   4: import json
   5: import platform
   6: import subprocess
   7: from pathlib import Path
   8: 
   9: import torch
  10: 
  11: from edge_opt.config import load_config
  12: from edge_opt.data import build_loaders
  13: from edge_opt.deploy import deployment_simulation
  14: from edge_opt.experiments import pareto_frontier, run_sweep, save_plots, train_model
  15: from edge_opt.hardware import estimate_layerwise_stats, precision_tradeoff_table, save_hardware_artifacts, summarize_hardware
  16: from edge_opt.metrics import collect_metrics, memory_violations
  17: from edge_opt.model import SmallCNN, resolve_device, set_deterministic
  18: from edge_opt.reporting import build_summary, write_outputs
  19: 
  20: 
  21: def _pip_freeze() -> list[str]:
  22:     try:
  23:         output = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
  24:     except Exception:
  25:         return []
  26:     return [line.strip() for line in output.splitlines() if line.strip()]
  27: 
  28: 
  29: def _git_commit_hash() -> str | None:
  30:     try:
  31:         return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  32:     except Exception:
  33:         return None
  34: 
  35: 
  36: def main() -> None:
  37:     parser = argparse.ArgumentParser(description="Hardware-aware edge AI optimization pipeline")
  38:     parser.add_argument("--config", type=str, default="configs/default.yaml")
  39:     args = parser.parse_args()
  40: 
  41:     cfg = load_config(args.config)
  42:     output_dir = Path(cfg.output_dir)
  43:     output_dir.mkdir(parents=True, exist_ok=True)
  44: 
  45:     set_deterministic(cfg.seed)
  46:     device = resolve_device(cfg.device)
  47:     latency_multiplier = 1.0 / max(cfg.cpu_frequency_scale, 1e-6)
  48: 
  49:     train_loader, val_loader = build_loaders(
  50:         dataset_name=cfg.dataset,
  51:         batch_size=cfg.batch_size,
  52:         train_subset=cfg.train_subset,
  53:         val_subset=cfg.val_subset,
  54:         seed=cfg.dataloader_seed,
  55:         num_workers=cfg.num_workers,
  56:     )
  57: 
  58:     baseline_model = SmallCNN()
  59:     baseline_model = train_model(
  60:         baseline_model,
  61:         train_loader,
  62:         epochs=cfg.epochs,
  63:         learning_rate=cfg.learning_rate,
  64:         device=device,
  65:     )
  66:     baseline_model.eval()
  67: 
  68:     baseline_metrics = collect_metrics(
  69:         baseline_model,
  70:         val_loader,
  71:         device=device,
  72:         power_watts=cfg.power_watts,
  73:         precision="fp32",
  74:         latency_multiplier=latency_multiplier,
  75:         benchmark_repeats=cfg.benchmark_repeats,
  76:         benchmark_trials=cfg.benchmark_trials,
  77:         benchmark_warmup=cfg.benchmark_warmup,
  78:     )
  79:     baseline_violations = memory_violations(baseline_metrics.model_memory_mb, cfg.memory_budgets_mb)
  80: 
  81:     sweep_df = run_sweep(
  82:         base_model=baseline_model,
  83:         train_loader=train_loader,
  84:         val_loader=val_loader,
  85:         calibration_loader=train_loader,
  86:         device=device,
  87:         pruning_levels=cfg.pruning_levels,
  88:         precisions=cfg.precisions,
  89:         power_watts=cfg.power_watts,
  90:         calibration_batches=cfg.calibration_batches,
  91:         memory_budgets_mb=cfg.memory_budgets_mb,
  92:         active_memory_budget_mb=cfg.active_memory_budget_mb,
  93:         latency_multiplier=latency_multiplier,
  94:         benchmark_repeats=cfg.benchmark_repeats,
  95:         benchmark_trials=cfg.benchmark_trials,
  96:         benchmark_warmup=cfg.benchmark_warmup,
  97:         fine_tune_epochs=cfg.fine_tune_epochs,
  98:         learning_rate=cfg.learning_rate,
  99:         quantization_backend=cfg.quantization_backend,
 100:         output_dir=output_dir,
 101:     )
 102: 
 103:     latency_frontier = pareto_frontier(sweep_df, x_col="latency_ms", use_ci=cfg.pareto_use_ci)
 104:     energy_frontier = pareto_frontier(sweep_df, x_col="energy_proxy_j", use_ci=cfg.pareto_use_ci)
 105:     save_plots(sweep_df, latency_frontier, energy_frontier, output_dir, show_error_bars=cfg.pareto_use_ci)
 106: 
 107:     layerwise_df = estimate_layerwise_stats(baseline_model, batch_size=cfg.batch_size)
 108:     hardware_summary = summarize_hardware(
 109:         layerwise_df,
 110:         latency_ms=baseline_metrics.latency_ms,
 111:         memory_bandwidth_gbps=cfg.memory_bandwidth_gbps,
 112:         peak_compute_gmacs=cfg.peak_compute_gmacs,
 113:     )
 114:     precision_df = precision_tradeoff_table(sweep_df)
 115:     save_hardware_artifacts(output_dir, layerwise_df, precision_df, hardware_summary)
 116: 
 117:     deploy_stats = deployment_simulation(baseline_model, val_loader, device=device, cpu_frequency_scale=cfg.cpu_frequency_scale)
 118: 
 119:     baseline_summary = {
 120:         **baseline_metrics.__dict__,
 121:         **baseline_violations,
 122:         "accepted_under_active_budget": baseline_metrics.model_memory_mb <= cfg.active_memory_budget_mb,
 123:     }
 124:     summary = build_summary(
 125:         baseline=baseline_summary,
 126:         memory_budgets_mb=cfg.memory_budgets_mb,
 127:         active_memory_budget_mb=cfg.active_memory_budget_mb,
 128:         cpu_frequency_scale=cfg.cpu_frequency_scale,
 129:         latency_multiplier=latency_multiplier,
 130:         sweep_df=sweep_df,
 131:         deployment={**deploy_stats, **hardware_summary},
 132:     )
 133: 
 134:     reproducibility = {
 135:         "packages": _pip_freeze(),
 136:         "cpu": {
 137:             "processor": platform.processor(),
 138:             "machine": platform.machine(),
 139:             "platform": platform.platform(),
 140:         },
 141:         "command": " ".join(["python", "scripts/run_pipeline.py", "--config", args.config]),
 142:         "git_commit_hash": _git_commit_hash(),
 143:         "seed": cfg.seed,
 144:         "dataloader_seed": cfg.dataloader_seed,
 145:         "device_requested": cfg.device,
 146:         "device_resolved": str(device),
 147:     }
 148: 
 149:     write_outputs(output_dir, sweep_df, latency_frontier, energy_frontier, summary)
 150:     (output_dir / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
 151: 
 152:     print(json.dumps(summary, indent=2))
 153: 
 154: 
 155: if __name__ == "__main__":
 156:     main()
```

## `src/edge_opt/__init__.py`

- Total lines: **1**
- Purpose: Core Python package module.

```text
   1: """Edge AI hardware optimization package."""
```

## `src/edge_opt/config.py`

- Total lines: **102**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: from dataclasses import dataclass
   4: from pathlib import Path
   5: from typing import Any
   6: 
   7: import yaml
   8: 
   9: 
  10: @dataclass
  11: class ExperimentConfig:
  12:     seed: int
  13:     dataset: str
  14:     batch_size: int
  15:     epochs: int
  16:     learning_rate: float
  17:     train_subset: int | None
  18:     val_subset: int | None
  19:     power_watts: float
  20:     pruning_levels: list[float]
  21:     precisions: list[str]
  22:     calibration_batches: int
  23:     output_dir: str
  24:     memory_budgets_mb: list[float]
  25:     active_memory_budget_mb: float
  26:     cpu_frequency_scale: float
  27:     dataloader_seed: int
  28:     num_workers: int
  29:     benchmark_repeats: int
  30:     memory_bandwidth_gbps: float
  31:     benchmark_trials: int
  32:     device: str = "cpu"
  33:     benchmark_warmup: int = 3
  34:     quantization_backend: str = "fbgemm"
  35:     fine_tune_epochs: int = 0
  36:     pareto_use_ci: bool = False
  37:     peak_compute_gmacs: float | None = None
  38: 
  39:     def __post_init__(self) -> None:
  40:         if self.batch_size <= 0 or self.epochs <= 0 or self.learning_rate <= 0:
  41:             raise ValueError("batch_size, epochs, and learning_rate must be > 0")
  42:         if any((level < 0.0 or level >= 1.0) for level in self.pruning_levels):
  43:             raise ValueError("all pruning levels must be in [0.0, 1.0)")
  44:         if not self.pruning_levels:
  45:             raise ValueError("pruning_levels must not be empty")
  46:         if not self.memory_budgets_mb or any(value <= 0 for value in self.memory_budgets_mb):
  47:             raise ValueError("memory_budgets_mb must contain positive values")
  48:         allowed_precisions = {"fp32", "fp16", "int8"}
  49:         if not self.precisions or any(p not in allowed_precisions for p in self.precisions):
  50:             raise ValueError("precisions must be non-empty and within {'fp32','fp16','int8'}")
  51:         if self.device not in {"cpu", "cuda", "mps"}:
  52:             raise ValueError("device must be one of {'cpu', 'cuda', 'mps'}")
  53:         if self.benchmark_warmup < 0:
  54:             raise ValueError("benchmark_warmup must be >= 0")
  55:         if self.benchmark_trials <= 0:
  56:             raise ValueError("benchmark_trials must be > 0")
  57:         if self.calibration_batches <= 0:
  58:             raise ValueError("calibration_batches must be > 0")
  59:         if self.fine_tune_epochs < 0:
  60:             raise ValueError("fine_tune_epochs must be >= 0")
  61:         if self.peak_compute_gmacs is not None and self.peak_compute_gmacs <= 0:
  62:             raise ValueError("peak_compute_gmacs must be > 0 when provided")
  63: 
  64: 
  65: def _require(raw: dict[str, Any], key: str) -> Any:
  66:     if key not in raw:
  67:         raise ValueError(f"Missing required config key '{key}'")
  68:     return raw[key]
  69: 
  70: 
  71: def load_config(path: str | Path) -> ExperimentConfig:
  72:     with open(path, "r", encoding="utf-8") as file:
  73:         raw = yaml.safe_load(file) or {}
  74: 
  75:     return ExperimentConfig(
  76:         seed=int(_require(raw, "seed")),
  77:         dataset=str(_require(raw, "dataset")),
  78:         batch_size=int(_require(raw, "batch_size")),
  79:         epochs=int(_require(raw, "epochs")),
  80:         learning_rate=float(_require(raw, "learning_rate")),
  81:         train_subset=int(raw["train_subset"]) if raw.get("train_subset") is not None else None,
  82:         val_subset=int(raw["val_subset"]) if raw.get("val_subset") is not None else None,
  83:         power_watts=float(_require(raw, "power_watts")),
  84:         pruning_levels=[float(v) for v in _require(raw, "pruning_levels")],
  85:         precisions=[str(v) for v in _require(raw, "precisions")],
  86:         calibration_batches=int(_require(raw, "calibration_batches")),
  87:         output_dir=str(_require(raw, "output_dir")),
  88:         memory_budgets_mb=[float(v) for v in _require(raw, "memory_budgets_mb")],
  89:         active_memory_budget_mb=float(_require(raw, "active_memory_budget_mb")),
  90:         cpu_frequency_scale=float(_require(raw, "cpu_frequency_scale")),
  91:         dataloader_seed=int(raw.get("dataloader_seed", _require(raw, "seed"))),
  92:         num_workers=int(raw.get("num_workers", 2)),
  93:         benchmark_repeats=int(raw.get("benchmark_repeats", 5)),
  94:         memory_bandwidth_gbps=float(raw.get("memory_bandwidth_gbps", 12.8)),
  95:         benchmark_trials=int(raw.get("benchmark_trials", 3)),
  96:         device=str(raw.get("device", "cpu")),
  97:         benchmark_warmup=int(raw.get("benchmark_warmup", 3)),
  98:         quantization_backend=str(raw.get("quantization_backend", "fbgemm")),
  99:         fine_tune_epochs=int(raw.get("fine_tune_epochs", 0)),
 100:         pareto_use_ci=bool(raw.get("pareto_use_ci", False)),
 101:         peak_compute_gmacs=float(raw["peak_compute_gmacs"]) if raw.get("peak_compute_gmacs") is not None else None,
 102:     )
```

## `src/edge_opt/data.py`

- Total lines: **43**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: import torch
   4: from torch.utils.data import DataLoader, Subset
   5: from torchvision import datasets, transforms
   6: 
   7: 
   8: DATASETS = {
   9:     "mnist": datasets.MNIST,
  10:     "fashion-mnist": datasets.FashionMNIST,
  11: }
  12: 
  13: 
  14: def build_loaders(
  15:     dataset_name: str,
  16:     batch_size: int,
  17:     train_subset: int | None,
  18:     val_subset: int | None,
  19:     seed: int = 42,
  20:     num_workers: int = 2,
  21: ) -> tuple[DataLoader, DataLoader]:
  22:     if dataset_name not in DATASETS:
  23:         msg = f"Unsupported dataset '{dataset_name}'. Use one of: {list(DATASETS)}"
  24:         raise ValueError(msg)
  25: 
  26:     transform = transforms.Compose([
  27:         transforms.ToTensor(),
  28:         transforms.Normalize((0.5,), (0.5,)),
  29:     ])
  30:     ds_cls = DATASETS[dataset_name]
  31: 
  32:     train_ds = ds_cls(root="data", train=True, download=True, transform=transform)
  33:     val_ds = ds_cls(root="data", train=False, download=True, transform=transform)
  34: 
  35:     if train_subset is not None:
  36:         train_ds = Subset(train_ds, list(range(min(train_subset, len(train_ds)))))
  37:     if val_subset is not None:
  38:         val_ds = Subset(val_ds, list(range(min(val_subset, len(val_ds)))))
  39: 
  40:     generator = torch.Generator().manual_seed(seed)
  41:     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
  42:     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  43:     return train_loader, val_loader
```

## `src/edge_opt/deploy.py`

- Total lines: **43**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: import time
   4: 
   5: import torch
   6: from torch import nn
   7: from torch.utils.data import DataLoader
   8: 
   9: 
  10: def deployment_simulation(
  11:     model: nn.Module,
  12:     loader: DataLoader,
  13:     device: torch.device,
  14:     cpu_frequency_scale: float,
  15:     stream_items: int = 128,
  16: ) -> dict[str, float]:
  17:     if len(loader) == 0:
  18:         raise ValueError("No batches in loader; check dataset or split.")
  19: 
  20:     model.eval()
  21:     batch_inputs, _ = next(iter(loader))
  22:     batch_inputs = batch_inputs.to(device)
  23:     latency_multiplier = 1.0 / max(cpu_frequency_scale, 1e-6)
  24: 
  25:     with torch.no_grad():
  26:         start_batch = time.perf_counter()
  27:         _ = model(batch_inputs)
  28:         batch_time = (time.perf_counter() - start_batch) * latency_multiplier
  29: 
  30:         stream = batch_inputs[:stream_items]
  31:         start_stream = time.perf_counter()
  32:         for item in stream:
  33:             _ = model(item.unsqueeze(0))
  34:         stream_time = (time.perf_counter() - start_stream) * latency_multiplier
  35: 
  36:     return {
  37:         "cpu_frequency_scale": cpu_frequency_scale,
  38:         "latency_multiplier": latency_multiplier,
  39:         "batch_latency_ms": batch_time * 1000.0,
  40:         "batch_throughput_sps": batch_inputs.shape[0] / batch_time,
  41:         "stream_avg_latency_ms": (stream_time / stream.shape[0]) * 1000.0,
  42:         "stream_throughput_sps": stream.shape[0] / stream_time,
  43:     }
```

## `src/edge_opt/experiments.py`

- Total lines: **213**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: from copy import deepcopy
   4: from dataclasses import asdict
   5: from pathlib import Path
   6: 
   7: import matplotlib.pyplot as plt
   8: import pandas as pd
   9: import torch
  10: from torch import nn
  11: from torch.utils.data import DataLoader
  12: 
  13: from edge_opt.metrics import PerfMetrics, collect_metrics, memory_violations
  14: from edge_opt.pruning import prune_and_finetune, structured_channel_prune
  15: from edge_opt.quantization import to_fp16, to_int8
  16: 
  17: 
  18: def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> nn.Module:
  19:     model.train()
  20:     for inputs, targets in train_loader:
  21:         inputs = inputs.to(device)
  22:         targets = targets.to(device)
  23:         optimizer.zero_grad()
  24:         outputs = model(inputs)
  25:         loss = criterion(outputs, targets)
  26:         loss.backward()
  27:         optimizer.step()
  28:     return model
  29: 
  30: 
  31: def train_model(model: nn.Module, train_loader: DataLoader, epochs: int, learning_rate: float, device: torch.device) -> nn.Module:
  32:     model = model.to(device)
  33:     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  34:     criterion = nn.CrossEntropyLoss()
  35:     for _ in range(epochs):
  36:         train_one_epoch(model, train_loader, optimizer, criterion, device)
  37:     return model
  38: 
  39: 
  40: def precision_variant(
  41:     model: nn.Module,
  42:     precision: str,
  43:     calibration_loader: DataLoader,
  44:     calibration_batches: int,
  45:     quantization_backend: str,
  46:     quant_metadata_path: Path | None,
  47: ) -> tuple[nn.Module, str]:
  48:     if precision == "fp32":
  49:         return deepcopy(model).eval(), "fp32"
  50:     if precision == "fp16":
  51:         return to_fp16(model), "fp16"
  52:     if precision == "int8":
  53:         return to_int8(
  54:             model,
  55:             calibration_loader,
  56:             calibration_batches=calibration_batches,
  57:             backend=quantization_backend,
  58:             metadata_path=quant_metadata_path,
  59:         ), "fp32"
  60:     msg = f"Unsupported precision '{precision}'"
  61:     raise ValueError(msg)
  62: 
  63: 
  64: def run_sweep(
  65:     base_model: nn.Module,
  66:     train_loader: DataLoader,
  67:     val_loader: DataLoader,
  68:     calibration_loader: DataLoader,
  69:     device: torch.device,
  70:     pruning_levels: list[float],
  71:     precisions: list[str],
  72:     power_watts: float,
  73:     calibration_batches: int,
  74:     memory_budgets_mb: list[float],
  75:     active_memory_budget_mb: float,
  76:     latency_multiplier: float,
  77:     benchmark_repeats: int = 5,
  78:     benchmark_trials: int = 3,
  79:     benchmark_warmup: int = 3,
  80:     fine_tune_epochs: int = 0,
  81:     learning_rate: float = 1e-3,
  82:     quantization_backend: str = "fbgemm",
  83:     output_dir: Path | None = None,
  84: ) -> pd.DataFrame:
  85:     rows: list[dict] = []
  86: 
  87:     for pruning in pruning_levels:
  88:         if fine_tune_epochs > 0:
  89:             optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
  90:             criterion = nn.CrossEntropyLoss()
  91: 
  92:             def _epoch(model: nn.Module, loader: DataLoader) -> nn.Module:
  93:                 return train_one_epoch(model, loader, optimizer, criterion, device)
  94: 
  95:             candidate = prune_and_finetune(base_model, pruning, fine_tune_epochs, train_loader, _epoch)
  96:         else:
  97:             candidate = structured_channel_prune(base_model, pruning)
  98:         candidate = candidate.to(device)
  99: 
 100:         for precision in precisions:
 101:             try:
 102:                 metadata_path = None
 103:                 if output_dir is not None and precision == "int8":
 104:                     metadata_path = output_dir / f"quantization_metadata_p{pruning}.json"
 105:                 variant, metric_precision = precision_variant(
 106:                     candidate,
 107:                     precision,
 108:                     calibration_loader,
 109:                     calibration_batches,
 110:                     quantization_backend,
 111:                     metadata_path,
 112:                 )
 113:                 variant = variant.to(device)
 114:                 metrics: PerfMetrics = collect_metrics(
 115:                     variant,
 116:                     val_loader,
 117:                     device,
 118:                     power_watts=power_watts,
 119:                     precision=metric_precision,
 120:                     latency_multiplier=latency_multiplier,
 121:                     benchmark_repeats=benchmark_repeats,
 122:                     benchmark_trials=benchmark_trials,
 123:                     benchmark_warmup=benchmark_warmup,
 124:                 )
 125:                 violations = memory_violations(metrics.model_memory_mb, memory_budgets_mb)
 126:                 rejected = metrics.model_memory_mb > active_memory_budget_mb
 127:                 row = {
 128:                     "pruning_level": pruning,
 129:                     "precision": precision,
 130:                     "accepted": not rejected,
 131:                     "error": None,
 132:                     "active_budget_mb": active_memory_budget_mb,
 133:                     **asdict(metrics),
 134:                     **violations,
 135:                 }
 136:             except Exception as exc:  # defensive to preserve sweep continuity
 137:                 row = {
 138:                     "pruning_level": pruning,
 139:                     "precision": precision,
 140:                     "accepted": False,
 141:                     "error": str(exc),
 142:                     "active_budget_mb": active_memory_budget_mb,
 143:                 }
 144:             rows.append(row)
 145: 
 146:     return pd.DataFrame(rows)
 147: 
 148: 
 149: def pareto_frontier(df: pd.DataFrame, x_col: str, use_ci: bool = False) -> pd.DataFrame:
 150:     ranked = df[df["accepted"]].copy()
 151:     acc_col = "accuracy_ci95_low" if use_ci and "accuracy_ci95_low" in ranked.columns else "accuracy"
 152:     ranked = ranked.sort_values([x_col, acc_col], ascending=[True, False]).reset_index(drop=True)
 153:     frontier = []
 154:     best_accuracy = -1.0
 155:     for _, row in ranked.iterrows():
 156:         if row[acc_col] > best_accuracy:
 157:             frontier.append(row)
 158:             best_accuracy = row[acc_col]
 159:     return pd.DataFrame(frontier)
 160: 
 161: 
 162: def save_plots(
 163:     df: pd.DataFrame,
 164:     latency_frontier: pd.DataFrame,
 165:     energy_frontier: pd.DataFrame,
 166:     output_dir: Path,
 167:     show_error_bars: bool = False,
 168: ) -> None:
 169:     output_dir.mkdir(parents=True, exist_ok=True)
 170:     accepted = df[df["accepted"]]
 171:     rejected = df[~df["accepted"]]
 172: 
 173:     plt.figure(figsize=(7, 5))
 174:     plt.scatter(accepted["latency_ms"], accepted["accuracy"], c="tab:blue", alpha=0.8, label="Accepted")
 175:     if show_error_bars and "accuracy_std" in accepted.columns:
 176:         plt.errorbar(accepted["latency_ms"], accepted["accuracy"], yerr=accepted["accuracy_std"], fmt="none", ecolor="tab:blue", alpha=0.35)
 177:     if not rejected.empty:
 178:         plt.scatter(rejected["latency_ms"], rejected["accuracy"], c="tab:gray", alpha=0.5, marker="x", label="Rejected")
 179:     plt.plot(latency_frontier["latency_ms"], latency_frontier["accuracy"], color="red", linewidth=2, label="Pareto")
 180:     plt.xlabel("Latency (ms)")
 181:     plt.ylabel("Accuracy")
 182:     plt.title("Accuracy vs Latency")
 183:     plt.legend()
 184:     plt.tight_layout()
 185:     plt.savefig(output_dir / "accuracy_vs_latency.png", dpi=180)
 186:     plt.close()
 187: 
 188:     plt.figure(figsize=(7, 5))
 189:     plt.scatter(accepted["energy_proxy_j"], accepted["accuracy"], c="tab:green", alpha=0.8, label="Accepted")
 190:     if show_error_bars and "accuracy_std" in accepted.columns:
 191:         plt.errorbar(accepted["energy_proxy_j"], accepted["accuracy"], yerr=accepted["accuracy_std"], fmt="none", ecolor="tab:green", alpha=0.35)
 192:     if not rejected.empty:
 193:         plt.scatter(rejected["energy_proxy_j"], rejected["accuracy"], c="tab:gray", alpha=0.5, marker="x", label="Rejected")
 194:     plt.plot(energy_frontier["energy_proxy_j"], energy_frontier["accuracy"], color="red", linewidth=2, label="Pareto")
 195:     plt.xlabel("Energy Proxy (J)")
 196:     plt.ylabel("Accuracy")
 197:     plt.title("Accuracy vs Energy")
 198:     plt.legend()
 199:     plt.tight_layout()
 200:     plt.savefig(output_dir / "accuracy_vs_energy.png", dpi=180)
 201:     plt.close()
 202: 
 203:     plt.figure(figsize=(7, 5))
 204:     plt.scatter(accepted["model_memory_mb"], accepted["accuracy"], c="tab:purple", alpha=0.8, label="Accepted")
 205:     if not rejected.empty:
 206:         plt.scatter(rejected["model_memory_mb"], rejected["accuracy"], c="tab:gray", alpha=0.5, marker="x", label="Rejected")
 207:     plt.xlabel("Model Memory (MB)")
 208:     plt.ylabel("Accuracy")
 209:     plt.title("Accuracy vs Memory")
 210:     plt.legend()
 211:     plt.tight_layout()
 212:     plt.savefig(output_dir / "accuracy_vs_memory.png", dpi=180)
 213:     plt.close()
```

## `src/edge_opt/hardware.py`

- Total lines: **171**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: from dataclasses import dataclass
   4: from pathlib import Path
   5: 
   6: import matplotlib.pyplot as plt
   7: import pandas as pd
   8: import torch
   9: from torch import nn
  10: 
  11: 
  12: @dataclass
  13: class LayerEstimate:
  14:     layer: str
  15:     output_elements: int
  16:     parameter_bytes: int
  17:     activation_bytes: int
  18:     macs: int
  19: 
  20: 
  21: def _module_parameter_bytes(module: nn.Module) -> int:
  22:     return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters(recurse=False))
  23: 
  24: 
  25: def _module_macs(module: nn.Module, output: torch.Tensor) -> int:
  26:     if isinstance(module, nn.Conv2d):
  27:         output_elements = output.numel()
  28:         kernel_height, kernel_width = module.kernel_size
  29:         groups = max(module.groups, 1)
  30:         in_channels_per_group = module.in_channels // groups
  31:         return int(output_elements * in_channels_per_group * kernel_height * kernel_width)
  32: 
  33:     if isinstance(module, nn.Linear):
  34:         batch_size = output.shape[0] if output.ndim > 0 else 1
  35:         return int(batch_size * module.in_features * module.out_features)
  36: 
  37:     return 0
  38: 
  39: 
  40: def estimate_layerwise_stats(
  41:     model: nn.Module,
  42:     batch_size: int,
  43:     input_shape: tuple[int, int, int] = (1, 28, 28),
  44:     activation_bytes_per_value: int | None = None,
  45: ) -> pd.DataFrame:
  46:     first_parameter = next(model.parameters(), None)
  47:     if first_parameter is None:
  48:         raise ValueError("Model has no parameters to profile")
  49: 
  50:     activation_size = activation_bytes_per_value or first_parameter.element_size()
  51:     tracked_modules: list[tuple[str, nn.Module]] = []
  52:     for name, module in model.named_modules():
  53:         if isinstance(module, (nn.Conv2d, nn.Linear)):
  54:             tracked_modules.append((name, module))
  55: 
  56:     if not tracked_modules:
  57:         raise ValueError("No supported layers found for profiling")
  58: 
  59:     layer_rows: list[LayerEstimate] = []
  60:     hooks = []
  61: 
  62:     def _make_hook(layer_name: str, layer_module: nn.Module):
  63:         def _hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
  64:             output_tensor = output if isinstance(output, torch.Tensor) else output[0]
  65:             output_elements = int(output_tensor.numel())
  66:             layer_rows.append(
  67:                 LayerEstimate(
  68:                     layer=layer_name,
  69:                     output_elements=output_elements,
  70:                     parameter_bytes=_module_parameter_bytes(layer_module),
  71:                     activation_bytes=output_elements * activation_size,
  72:                     macs=_module_macs(layer_module, output_tensor),
  73:                 )
  74:             )
  75: 
  76:         return _hook
  77: 
  78:     for layer_name, layer_module in tracked_modules:
  79:         hooks.append(layer_module.register_forward_hook(_make_hook(layer_name, layer_module)))
  80: 
  81:     try:
  82:         device = first_parameter.device
  83:         model_was_training = model.training
  84:         model.eval()
  85:         with torch.no_grad():
  86:             sample = torch.zeros((batch_size, *input_shape), device=device, dtype=first_parameter.dtype)
  87:             model(sample)
  88:         if model_was_training:
  89:             model.train()
  90:     finally:
  91:         for hook in hooks:
  92:             hook.remove()
  93: 
  94:     return pd.DataFrame([vars(row) for row in layer_rows])
  95: 
  96: 
  97: def summarize_hardware(
  98:     layerwise_df: pd.DataFrame,
  99:     latency_ms: float,
 100:     memory_bandwidth_gbps: float,
 101:     peak_compute_gmacs: float | None = None,
 102: ) -> dict[str, float | str]:
 103:     total_bytes = float(layerwise_df["parameter_bytes"].sum() + layerwise_df["activation_bytes"].sum())
 104:     total_macs = float(layerwise_df["macs"].sum())
 105:     latency_s = max(latency_ms / 1000.0, 1e-9)
 106:     achieved_bandwidth_gbps = (total_bytes / latency_s) / 1e9
 107:     bandwidth_utilization = achieved_bandwidth_gbps / max(memory_bandwidth_gbps, 1e-9)
 108:     achieved_gmacs = (total_macs / latency_s) / 1e9
 109:     arithmetic_intensity = total_macs / max(total_bytes, 1.0)
 110: 
 111:     summary: dict[str, float | str] = {
 112:         "estimated_total_bytes": total_bytes,
 113:         "estimated_total_macs": total_macs,
 114:         "achieved_bandwidth_gbps": achieved_bandwidth_gbps,
 115:         "configured_memory_bandwidth_gbps": memory_bandwidth_gbps,
 116:         "bandwidth_utilization": bandwidth_utilization,
 117:         "achieved_gmacs": achieved_gmacs,
 118:         "arithmetic_intensity_macs_per_byte": arithmetic_intensity,
 119:     }
 120: 
 121:     if peak_compute_gmacs is not None:
 122:         roofline_knee = peak_compute_gmacs / max(memory_bandwidth_gbps, 1e-9)
 123:         bound = "memory-bound" if arithmetic_intensity < roofline_knee else "compute-bound"
 124:         summary.update(
 125:             {
 126:                 "configured_peak_compute_gmacs": peak_compute_gmacs,
 127:                 "roofline_knee_macs_per_byte": roofline_knee,
 128:                 "bound_regime": bound,
 129:             }
 130:         )
 131: 
 132:     return summary
 133: 
 134: 
 135: def precision_tradeoff_table(sweep_df: pd.DataFrame) -> pd.DataFrame:
 136:     grouped = sweep_df.groupby("precision", as_index=False).agg(
 137:         accuracy_mean=("accuracy", "mean"),
 138:         latency_ms_mean=("latency_ms", "mean"),
 139:         memory_mb_mean=("memory_mb", "mean"),
 140:         energy_proxy_j_mean=("energy_proxy_j", "mean"),
 141:         accepted_ratio=("accepted", "mean"),
 142:     )
 143:     return grouped.sort_values("latency_ms_mean").reset_index(drop=True)
 144: 
 145: 
 146: def save_hardware_artifacts(
 147:     output_dir: Path,
 148:     layerwise_df: pd.DataFrame,
 149:     precision_df: pd.DataFrame,
 150:     summary: dict[str, float | str],
 151: ) -> None:
 152:     output_dir.mkdir(parents=True, exist_ok=True)
 153:     layerwise_df.to_csv(output_dir / "layerwise_breakdown.csv", index=False)
 154:     precision_df.to_csv(output_dir / "precision_tradeoffs.csv", index=False)
 155:     pd.DataFrame([summary]).to_csv(output_dir / "hardware_summary.csv", index=False)
 156: 
 157:     plt.figure(figsize=(7, 5))
 158:     plt.bar(layerwise_df["layer"], layerwise_df["activation_bytes"] / (1024**2), color="tab:orange")
 159:     plt.ylabel("Activation Memory (MB)")
 160:     plt.title("Layer-wise Activation Memory")
 161:     plt.tight_layout()
 162:     plt.savefig(output_dir / "layerwise_activation_memory.png", dpi=180)
 163:     plt.close()
 164: 
 165:     plt.figure(figsize=(7, 5))
 166:     plt.bar(layerwise_df["layer"], layerwise_df["macs"] / 1e6, color="tab:blue")
 167:     plt.ylabel("MACs (Millions)")
 168:     plt.title("Layer-wise Compute Estimate")
 169:     plt.tight_layout()
 170:     plt.savefig(output_dir / "layerwise_macs.png", dpi=180)
 171:     plt.close()
```

## `src/edge_opt/metrics.py`

- Total lines: **179**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: import time
   4: from dataclasses import dataclass
   5: 
   6: import torch
   7: from torch import nn
   8: from torch.utils import benchmark
   9: from torch.utils.data import DataLoader
  10: 
  11: 
  12: @dataclass
  13: class PerfMetrics:
  14:     accuracy: float
  15:     accuracy_std: float
  16:     accuracy_ci95_low: float
  17:     accuracy_ci95_high: float
  18:     latency_ms: float
  19:     latency_median_ms: float
  20:     latency_std_ms: float
  21:     latency_p95_ms: float
  22:     throughput_sps: float
  23:     model_memory_mb: float
  24:     memory_mb: float
  25:     estimated_runtime_memory_mb: float
  26:     energy_proxy_j: float
  27:     energy_proxy_note: str
  28: 
  29: 
  30: def _sync_device(device: torch.device) -> None:
  31:     if device.type == "cuda":
  32:         torch.cuda.synchronize(device)
  33:     elif device.type == "mps":
  34:         torch.mps.synchronize()
  35: 
  36: 
  37: def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, precision: str = "fp32") -> float:
  38:     model.eval()
  39:     total = 0
  40:     correct = 0
  41:     with torch.no_grad():
  42:         for inputs, targets in loader:
  43:             inputs = inputs.to(device)
  44:             targets = targets.to(device)
  45:             if precision == "fp16":
  46:                 inputs = inputs.half()
  47:             outputs = model(inputs)
  48:             pred = outputs.argmax(dim=1)
  49:             total += targets.size(0)
  50:             correct += (pred == targets).sum().item()
  51:     return correct / total
  52: 
  53: 
  54: def evaluate_accuracy_distribution(
  55:     model: nn.Module,
  56:     loader: DataLoader,
  57:     device: torch.device,
  58:     precision: str,
  59:     trials: int,
  60: ) -> tuple[float, float, float, float]:
  61:     accuracies = [evaluate_accuracy(model, loader, device, precision=precision) for _ in range(trials)]
  62:     tensor = torch.tensor(accuracies, dtype=torch.float32)
  63:     mean = float(tensor.mean())
  64:     std = float(tensor.std(unbiased=False))
  65:     ci_half_width = float(1.96 * (std / max(trials**0.5, 1.0)))
  66:     return mean, std, max(0.0, mean - ci_half_width), min(1.0, mean + ci_half_width)
  67: 
  68: 
  69: def measure_latency(model: nn.Module, sample_input: torch.Tensor, device: torch.device, num_runs: int = 100, warmup: int = 3) -> float:
  70:     model.eval()
  71:     with torch.no_grad():
  72:         for _ in range(warmup):
  73:             _ = model(sample_input)
  74:         _sync_device(device)
  75: 
  76:         try:
  77:             timer = benchmark.Timer(stmt="model(sample_input)", globals={"model": model, "sample_input": sample_input})
  78:             result = timer.blocked_autorange(min_run_time=max(0.1, num_runs * 0.001))
  79:             return result.median * 1000.0
  80:         except Exception:
  81:             start = time.perf_counter()
  82:             for _ in range(num_runs):
  83:                 _ = model(sample_input)
  84:             _sync_device(device)
  85:             elapsed = time.perf_counter() - start
  86:             return (elapsed / num_runs) * 1000.0
  87: 
  88: 
  89: def measure_latency_distribution(
  90:     model: nn.Module,
  91:     sample_input: torch.Tensor,
  92:     device: torch.device,
  93:     repeats: int = 5,
  94:     num_runs: int = 100,
  95:     warmup: int = 3,
  96: ) -> tuple[float, float, float, float]:
  97:     latencies = [measure_latency(model, sample_input, device=device, num_runs=num_runs, warmup=warmup) for _ in range(repeats)]
  98:     latency_tensor = torch.tensor(latencies, dtype=torch.float32)
  99:     return (
 100:         float(latency_tensor.mean()),
 101:         float(torch.median(latency_tensor)),
 102:         float(latency_tensor.std(unbiased=False)),
 103:         float(torch.quantile(latency_tensor, 0.95)),
 104:     )
 105: 
 106: 
 107: def model_memory_mb(model: nn.Module) -> float:
 108:     """Compute model parameter memory from the state dict only."""
 109:     total_bytes = 0
 110:     for tensor in model.state_dict().values():
 111:         if isinstance(tensor, torch.Tensor):
 112:             total_bytes += tensor.numel() * tensor.element_size()
 113:     return total_bytes / (1024**2)
 114: 
 115: 
 116: def estimated_runtime_memory_mb(parameter_memory_mb: float) -> float:
 117:     """Estimate runtime memory including activations as ~1.5x parameter memory."""
 118:     return parameter_memory_mb * 1.5
 119: 
 120: 
 121: def memory_violations(memory_mb: float, budgets_mb: list[float]) -> dict[str, bool]:
 122:     return {f"violates_{budget}mb": memory_mb > budget for budget in budgets_mb}
 123: 
 124: 
 125: def collect_metrics(
 126:     model: nn.Module,
 127:     loader: DataLoader,
 128:     device: torch.device,
 129:     power_watts: float,
 130:     precision: str,
 131:     latency_multiplier: float = 1.0,
 132:     benchmark_repeats: int = 5,
 133:     benchmark_trials: int = 3,
 134:     benchmark_warmup: int = 3,
 135: ) -> PerfMetrics:
 136:     if len(loader) == 0:
 137:         raise ValueError("No batches in loader; check dataset or split.")
 138: 
 139:     sample_batch, _ = next(iter(loader))
 140:     sample_input = sample_batch.to(device)
 141:     if precision == "fp16":
 142:         sample_input = sample_input.half()
 143: 
 144:     accuracy, accuracy_std, ci_low, ci_high = evaluate_accuracy_distribution(
 145:         model,
 146:         loader,
 147:         device,
 148:         precision=precision,
 149:         trials=benchmark_trials,
 150:     )
 151:     latency_mean, latency_median, latency_std, latency_p95 = measure_latency_distribution(
 152:         model,
 153:         sample_input,
 154:         device=device,
 155:         repeats=benchmark_repeats,
 156:         warmup=benchmark_warmup,
 157:     )
 158:     latency = latency_mean * latency_multiplier
 159:     throughput = sample_input.shape[0] / (latency / 1000.0)
 160:     param_memory = model_memory_mb(model)
 161:     runtime_memory = estimated_runtime_memory_mb(param_memory)
 162:     energy_proxy = (latency / 1000.0) * power_watts
 163: 
 164:     return PerfMetrics(
 165:         accuracy=accuracy,
 166:         accuracy_std=accuracy_std,
 167:         accuracy_ci95_low=ci_low,
 168:         accuracy_ci95_high=ci_high,
 169:         latency_ms=latency,
 170:         latency_median_ms=latency_median * latency_multiplier,
 171:         latency_std_ms=latency_std * latency_multiplier,
 172:         latency_p95_ms=latency_p95 * latency_multiplier,
 173:         throughput_sps=throughput,
 174:         model_memory_mb=param_memory,
 175:         memory_mb=param_memory,
 176:         estimated_runtime_memory_mb=runtime_memory,
 177:         energy_proxy_j=energy_proxy,
 178:         energy_proxy_note="Proxy metric computed as power_watts × latency_s (not measured on-device power)",
 179:     )
```

## `src/edge_opt/model.py`

- Total lines: **55**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: import random
   4: import warnings
   5: 
   6: import numpy as np
   7: import torch
   8: from torch import nn
   9: 
  10: 
  11: class SmallCNN(nn.Module):
  12:     def __init__(self, conv1_channels: int = 16, conv2_channels: int = 32, num_classes: int = 10) -> None:
  13:         super().__init__()
  14:         self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
  15:         self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
  16:         self.pool = nn.MaxPool2d(2)
  17:         self.relu = nn.ReLU(inplace=True)
  18:         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
  19:         self.classifier = nn.Linear(conv2_channels, num_classes)
  20: 
  21:     def forward(self, x: torch.Tensor) -> torch.Tensor:
  22:         x = self.pool(self.relu(self.conv1(x)))
  23:         x = self.pool(self.relu(self.conv2(x)))
  24:         x = self.global_pool(x)
  25:         x = x.flatten(start_dim=1)
  26:         return self.classifier(x)
  27: 
  28: 
  29: def resolve_device(device_name: str) -> torch.device:
  30:     """Resolve requested device with graceful fallback and warning."""
  31:     if device_name == "cpu":
  32:         return torch.device("cpu")
  33:     if device_name == "cuda":
  34:         if torch.cuda.is_available():
  35:             return torch.device("cuda")
  36:         warnings.warn("CUDA requested but unavailable; falling back to CPU.", stacklevel=2)
  37:         return torch.device("cpu")
  38:     if device_name == "mps":
  39:         if torch.backends.mps.is_available():
  40:             return torch.device("mps")
  41:         warnings.warn("MPS requested but unavailable; falling back to CPU.", stacklevel=2)
  42:         return torch.device("cpu")
  43: 
  44:     warnings.warn(f"Unknown device '{device_name}'; falling back to CPU.", stacklevel=2)
  45:     return torch.device("cpu")
  46: 
  47: 
  48: def set_deterministic(seed: int) -> None:
  49:     random.seed(seed)
  50:     np.random.seed(seed)
  51:     torch.manual_seed(seed)
  52:     torch.cuda.manual_seed_all(seed)
  53:     torch.backends.cudnn.deterministic = True
  54:     torch.backends.cudnn.benchmark = False
  55:     torch.use_deterministic_algorithms(True)
```

## `src/edge_opt/pruning.py`

- Total lines: **54**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: from collections.abc import Callable
   4: 
   5: import torch
   6: from torch.utils.data import DataLoader
   7: 
   8: from edge_opt.model import SmallCNN
   9: 
  10: 
  11: def _topk_indices(channel_scores: torch.Tensor, pruning_level: float) -> torch.Tensor:
  12:     total = channel_scores.numel()
  13:     keep = max(1, int(round(total * (1.0 - pruning_level))))
  14:     return torch.topk(channel_scores, keep, largest=True).indices.sort().values
  15: 
  16: 
  17: def structured_channel_prune(model: SmallCNN, pruning_level: float) -> SmallCNN:
  18:     if not 0.0 <= pruning_level < 1.0:
  19:         msg = "pruning_level must be in [0.0, 1.0)."
  20:         raise ValueError(msg)
  21: 
  22:     conv1_scores = model.conv1.weight.data.abs().sum(dim=(1, 2, 3))
  23:     keep1 = _topk_indices(conv1_scores, pruning_level)
  24: 
  25:     conv2_scores = model.conv2.weight.data.abs().sum(dim=(1, 2, 3))
  26:     keep2 = _topk_indices(conv2_scores, pruning_level)
  27: 
  28:     pruned = SmallCNN(conv1_channels=len(keep1), conv2_channels=len(keep2), num_classes=model.classifier.out_features)
  29: 
  30:     with torch.no_grad():
  31:         pruned.conv1.weight.copy_(model.conv1.weight[keep1])
  32:         pruned.conv1.bias.copy_(model.conv1.bias[keep1])
  33: 
  34:         conv2_w = model.conv2.weight[keep2][:, keep1, :, :]
  35:         pruned.conv2.weight.copy_(conv2_w)
  36:         pruned.conv2.bias.copy_(model.conv2.bias[keep2])
  37: 
  38:         pruned.classifier.weight.copy_(model.classifier.weight[:, keep2])
  39:         pruned.classifier.bias.copy_(model.classifier.bias)
  40: 
  41:     return pruned
  42: 
  43: 
  44: def prune_and_finetune(
  45:     model: SmallCNN,
  46:     pruning_level: float,
  47:     fine_tune_epochs: int,
  48:     train_loader: DataLoader,
  49:     train_one_epoch: Callable[[SmallCNN, DataLoader], SmallCNN],
  50: ) -> SmallCNN:
  51:     pruned = structured_channel_prune(model, pruning_level)
  52:     for _ in range(fine_tune_epochs):
  53:         pruned = train_one_epoch(pruned, train_loader)
  54:     return pruned
```

## `src/edge_opt/quantization.py`

- Total lines: **70**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: import json
   4: import warnings
   5: from copy import deepcopy
   6: from pathlib import Path
   7: 
   8: import torch
   9: from torch import nn
  10: from torch.ao.quantization import get_default_qconfig_mapping
  11: from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
  12: from torch.utils.data import DataLoader
  13: 
  14: 
  15: def _default_backend() -> str:
  16:     return "qnnpack" if "arm" in torch.backends.quantized.engine.lower() else "fbgemm"
  17: 
  18: 
  19: def to_fp16(model: nn.Module) -> nn.Module:
  20:     fp16_model = deepcopy(model).half().eval()
  21:     return fp16_model
  22: 
  23: 
  24: def to_int8(
  25:     model: nn.Module,
  26:     calibration_loader: DataLoader,
  27:     calibration_batches: int = 10,
  28:     backend: str | None = None,
  29:     metadata_path: str | Path | None = None,
  30: ) -> nn.Module:
  31:     backend_name = backend or _default_backend()
  32:     float_model = deepcopy(model).eval().to("cpu")
  33: 
  34:     try:
  35:         qconfig_mapping = get_default_qconfig_mapping(backend_name)
  36:         example_inputs, _ = next(iter(calibration_loader))
  37:         prepared = prepare_fx(float_model, qconfig_mapping, example_inputs=(example_inputs.cpu(),))
  38: 
  39:         with torch.no_grad():
  40:             for index, (inputs, _) in enumerate(calibration_loader):
  41:                 _ = prepared(inputs.cpu())
  42:                 if index + 1 >= calibration_batches:
  43:                     break
  44: 
  45:         quantized = convert_fx(prepared)
  46:     except Exception as exc:
  47:         warnings.warn(f"INT8 quantization backend '{backend_name}' failed ({exc}); using CPU float model.", stacklevel=2)
  48:         quantized = float_model
  49: 
  50:     if metadata_path is not None:
  51:         metadata = {
  52:             "backend": backend_name,
  53:             "calibration_batches": calibration_batches,
  54:             "quantized": quantized is not float_model,
  55:             "scale": None,
  56:             "zero_point": None,
  57:         }
  58:         for _, module in quantized.named_modules():
  59:             scale = getattr(module, "scale", None)
  60:             zero_point = getattr(module, "zero_point", None)
  61:             if scale is not None and zero_point is not None:
  62:                 metadata["scale"] = float(scale)
  63:                 metadata["zero_point"] = int(zero_point)
  64:                 break
  65: 
  66:         path = Path(metadata_path)
  67:         path.parent.mkdir(parents=True, exist_ok=True)
  68:         path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  69: 
  70:     return quantized
```

## `src/edge_opt/reporting.py`

- Total lines: **47**
- Purpose: Core Python package module.

```text
   1: from __future__ import annotations
   2: 
   3: import json
   4: from pathlib import Path
   5: from typing import Any
   6: 
   7: import pandas as pd
   8: 
   9: 
  10: def build_summary(
  11:     baseline: dict[str, Any],
  12:     memory_budgets_mb: list[float],
  13:     active_memory_budget_mb: float,
  14:     cpu_frequency_scale: float,
  15:     latency_multiplier: float,
  16:     sweep_df: pd.DataFrame,
  17:     deployment: dict[str, float],
  18: ) -> dict[str, Any]:
  19:     accepted = sweep_df[sweep_df["accepted"]]
  20:     return {
  21:         "baseline": baseline,
  22:         "memory_budgets_mb": memory_budgets_mb,
  23:         "active_memory_budget_mb": active_memory_budget_mb,
  24:         "cpu_frequency_scale": cpu_frequency_scale,
  25:         "latency_multiplier": latency_multiplier,
  26:         "study_rows": len(sweep_df),
  27:         "accepted_rows": int(sweep_df["accepted"].sum()),
  28:         "rejected_rows": int((~sweep_df["accepted"]).sum()),
  29:         "best_accuracy_accepted": float(accepted["accuracy"].max()) if not accepted.empty else None,
  30:         "lowest_latency_ms_accepted": float(accepted["latency_ms"].min()) if not accepted.empty else None,
  31:         "deployment": deployment,
  32:     }
  33: 
  34: 
  35: def write_outputs(
  36:     output_dir: Path,
  37:     sweep_df: pd.DataFrame,
  38:     latency_frontier: pd.DataFrame,
  39:     energy_frontier: pd.DataFrame,
  40:     summary: dict[str, Any],
  41: ) -> None:
  42:     output_dir.mkdir(parents=True, exist_ok=True)
  43:     sweep_df.to_csv(output_dir / "sweep_results.csv", index=False)
  44:     latency_frontier.to_csv(output_dir / "pareto_frontier_latency.csv", index=False)
  45:     energy_frontier.to_csv(output_dir / "pareto_frontier_energy.csv", index=False)
  46:     with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
  47:         json.dump(summary, file, indent=2)
```

## `tests/baseline_metrics.json`

- Total lines: **4**
- Purpose: Automated test coverage and fixtures.

```text
   1: {
   2:   "accuracy": 0.125,
   3:   "model_memory_mb": 0.02356719970703125
   4: }
```

## `tests/conftest.py`

- Total lines: **10**
- Purpose: Automated test coverage and fixtures.

```text
   1: from __future__ import annotations
   2: 
   3: import sys
   4: from pathlib import Path
   5: 
   6: ROOT = Path(__file__).resolve().parents[1]
   7: SRC = ROOT / "src"
   8: for path in (str(ROOT), str(SRC)):
   9:     if path not in sys.path:
  10:         sys.path.insert(0, path)
```

## `tests/test_config.py`

- Total lines: **93**
- Purpose: Automated test coverage and fixtures.

```text
   1: from pathlib import Path
   2: 
   3: import pytest
   4: 
   5: from edge_opt.config import ExperimentConfig, load_config
   6: 
   7: 
   8: BASE = """
   9: seed: 1
  10: dataset: fashion-mnist
  11: batch_size: 32
  12: epochs: 1
  13: learning_rate: 0.001
  14: power_watts: 3.0
  15: pruning_levels: [0.0, 0.5]
  16: precisions: [fp32, fp16]
  17: calibration_batches: 2
  18: output_dir: outputs
  19: memory_budgets_mb: [1.0]
  20: active_memory_budget_mb: 1.0
  21: cpu_frequency_scale: 0.8
  22: """.strip()
  23: 
  24: 
  25: def test_load_config_with_yaml_parser_and_validation(tmp_path: Path) -> None:
  26:     config_path = tmp_path / "config.yaml"
  27:     config_path.write_text(BASE)
  28: 
  29:     cfg = load_config(config_path)
  30:     assert cfg.dataloader_seed == 1
  31:     assert cfg.benchmark_trials == 3
  32:     assert cfg.device == "cpu"
  33: 
  34: 
  35: def test_load_config_rejects_invalid_pruning(tmp_path: Path) -> None:
  36:     config_path = tmp_path / "config.yaml"
  37:     config_path.write_text(BASE.replace("[0.0, 0.5]", "[1.2]"))
  38: 
  39:     with pytest.raises(ValueError):
  40:         load_config(config_path)
  41: 
  42: 
  43: def test_config_rejects_zero_trials() -> None:
  44:     with pytest.raises(ValueError):
  45:         ExperimentConfig(
  46:             seed=1,
  47:             dataset="fashion-mnist",
  48:             batch_size=32,
  49:             epochs=1,
  50:             learning_rate=0.001,
  51:             train_subset=None,
  52:             val_subset=None,
  53:             power_watts=2.0,
  54:             pruning_levels=[0.0],
  55:             precisions=["fp32"],
  56:             calibration_batches=1,
  57:             output_dir="outputs",
  58:             memory_budgets_mb=[1.0],
  59:             active_memory_budget_mb=1.0,
  60:             cpu_frequency_scale=1.0,
  61:             dataloader_seed=1,
  62:             num_workers=0,
  63:             benchmark_repeats=1,
  64:             memory_bandwidth_gbps=10.0,
  65:             benchmark_trials=0,
  66:         )
  67: 
  68: 
  69: def test_config_rejects_non_positive_peak_compute() -> None:
  70:     with pytest.raises(ValueError):
  71:         ExperimentConfig(
  72:             seed=1,
  73:             dataset="fashion-mnist",
  74:             batch_size=32,
  75:             epochs=1,
  76:             learning_rate=0.001,
  77:             train_subset=None,
  78:             val_subset=None,
  79:             power_watts=2.0,
  80:             pruning_levels=[0.0],
  81:             precisions=["fp32"],
  82:             calibration_batches=1,
  83:             output_dir="outputs",
  84:             memory_budgets_mb=[1.0],
  85:             active_memory_budget_mb=1.0,
  86:             cpu_frequency_scale=1.0,
  87:             dataloader_seed=1,
  88:             num_workers=0,
  89:             benchmark_repeats=1,
  90:             memory_bandwidth_gbps=10.0,
  91:             benchmark_trials=1,
  92:             peak_compute_gmacs=0.0,
  93:         )
```

## `tests/test_edge_cases.py`

- Total lines: **66**
- Purpose: Automated test coverage and fixtures.

```text
   1: from pathlib import Path
   2: 
   3: import pytest
   4: import torch
   5: from torch.utils.data import DataLoader, TensorDataset
   6: 
   7: from edge_opt.config import ExperimentConfig
   8: from edge_opt.deploy import deployment_simulation
   9: from edge_opt.model import SmallCNN, resolve_device
  10: from edge_opt.quantization import to_int8
  11: 
  12: 
  13: def test_deployment_simulation_rejects_empty_loader() -> None:
  14:     x = torch.randn(0, 1, 28, 28)
  15:     y = torch.randint(0, 10, (0,))
  16:     loader = DataLoader(TensorDataset(x, y), batch_size=4)
  17:     with pytest.raises(ValueError, match="No batches"):
  18:         deployment_simulation(SmallCNN(), loader, device=torch.device("cpu"), cpu_frequency_scale=1.0)
  19: 
  20: 
  21: def test_invalid_device_string_warns_and_falls_back() -> None:
  22:     with pytest.warns(UserWarning):
  23:         device = resolve_device("invalid_device")
  24:     assert str(device) == "cpu"
  25: 
  26: 
  27: def test_quantization_backend_failure_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  28:     x = torch.randn(8, 1, 28, 28)
  29:     y = torch.randint(0, 10, (8,))
  30:     loader = DataLoader(TensorDataset(x, y), batch_size=4)
  31: 
  32:     def _raise(*args, **kwargs):
  33:         raise RuntimeError("backend boom")
  34: 
  35:     monkeypatch.setattr("edge_opt.quantization.get_default_qconfig_mapping", _raise)
  36: 
  37:     with pytest.warns(UserWarning):
  38:         quantized = to_int8(SmallCNN(), loader, calibration_batches=1, backend="fbgemm", metadata_path=tmp_path / "q.json")
  39:     assert isinstance(quantized, SmallCNN)
  40:     assert (tmp_path / "q.json").exists()
  41: 
  42: 
  43: def test_invalid_config_values_raise() -> None:
  44:     with pytest.raises(ValueError):
  45:         ExperimentConfig(
  46:             seed=1,
  47:             dataset="fashion-mnist",
  48:             batch_size=-1,
  49:             epochs=1,
  50:             learning_rate=0.001,
  51:             train_subset=None,
  52:             val_subset=None,
  53:             power_watts=2.0,
  54:             pruning_levels=[0.0],
  55:             precisions=["fp32"],
  56:             calibration_batches=1,
  57:             output_dir="outputs",
  58:             memory_budgets_mb=[1.0],
  59:             active_memory_budget_mb=1.0,
  60:             cpu_frequency_scale=1.0,
  61:             dataloader_seed=1,
  62:             num_workers=0,
  63:             benchmark_repeats=1,
  64:             memory_bandwidth_gbps=10.0,
  65:             benchmark_trials=1,
  66:         )
```

## `tests/test_experiments.py`

- Total lines: **51**
- Purpose: Automated test coverage and fixtures.

```text
   1: from pathlib import Path
   2: 
   3: import pandas as pd
   4: import torch
   5: from torch.utils.data import DataLoader, TensorDataset
   6: 
   7: from edge_opt.experiments import pareto_frontier, run_sweep
   8: from edge_opt.model import SmallCNN
   9: 
  10: 
  11: def _dummy_loader() -> DataLoader:
  12:     x = torch.randn(16, 1, 28, 28)
  13:     y = torch.randint(0, 10, (16,))
  14:     return DataLoader(TensorDataset(x, y), batch_size=8)
  15: 
  16: 
  17: def test_pareto_frontier_filters_dominated_points() -> None:
  18:     df = pd.DataFrame(
  19:         [
  20:             {"accepted": True, "latency_ms": 10.0, "accuracy": 0.80, "accuracy_ci95_low": 0.78},
  21:             {"accepted": True, "latency_ms": 8.0, "accuracy": 0.78, "accuracy_ci95_low": 0.76},
  22:             {"accepted": True, "latency_ms": 12.0, "accuracy": 0.79, "accuracy_ci95_low": 0.70},
  23:         ]
  24:     )
  25:     frontier = pareto_frontier(df, "latency_ms", use_ci=True)
  26:     assert list(frontier["latency_ms"]) == [8.0, 10.0]
  27: 
  28: 
  29: def test_run_sweep_captures_variant_errors() -> None:
  30:     loader = _dummy_loader()
  31:     model = SmallCNN()
  32:     out = run_sweep(
  33:         base_model=model,
  34:         train_loader=loader,
  35:         val_loader=loader,
  36:         calibration_loader=loader,
  37:         device=torch.device("cpu"),
  38:         pruning_levels=[0.0],
  39:         precisions=["fp32", "bad_precision"],
  40:         power_watts=2.0,
  41:         calibration_batches=1,
  42:         memory_budgets_mb=[1.0],
  43:         active_memory_budget_mb=2.0,
  44:         latency_multiplier=1.0,
  45:         benchmark_repeats=1,
  46:         benchmark_trials=1,
  47:         output_dir=Path("outputs/test"),
  48:     )
  49:     assert len(out) == 2
  50:     assert out["error"].isna().sum() == 1
  51:     assert out["error"].notna().sum() == 1
```

## `tests/test_integration_cli.py`

- Total lines: **57**
- Purpose: Automated test coverage and fixtures.

```text
   1: from __future__ import annotations
   2: 
   3: import sys
   4: from pathlib import Path
   5: 
   6: import torch
   7: from torch.utils.data import DataLoader, TensorDataset
   8: 
   9: from scripts.run_pipeline import main
  10: 
  11: 
  12: def test_pipeline_cli_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
  13:     cfg = tmp_path / "config.yaml"
  14:     cfg.write_text(
  15:         f"""
  16: seed: 1
  17: dataset: fashion-mnist
  18: batch_size: 4
  19: epochs: 1
  20: learning_rate: 0.001
  21: train_subset: 8
  22: val_subset: 8
  23: power_watts: 2.0
  24: pruning_levels: [0.0]
  25: precisions: [fp32]
  26: calibration_batches: 1
  27: output_dir: {tmp_path / 'out'}
  28: memory_budgets_mb: [10.0]
  29: active_memory_budget_mb: 10.0
  30: cpu_frequency_scale: 1.0
  31: benchmark_repeats: 1
  32: benchmark_trials: 1
  33: benchmark_warmup: 0
  34: num_workers: 0
  35: device: cpu
  36: quantization_backend: fbgemm
  37: fine_tune_epochs: 0
  38: pareto_use_ci: false
  39: """.strip(),
  40:         encoding="utf-8",
  41:     )
  42: 
  43:     def _fake_build_loaders(*args, **kwargs):
  44:         x = torch.randn(8, 1, 28, 28)
  45:         y = torch.randint(0, 10, (8,))
  46:         loader = DataLoader(TensorDataset(x, y), batch_size=4)
  47:         return loader, loader
  48: 
  49:     monkeypatch.setattr("scripts.run_pipeline.build_loaders", _fake_build_loaders)
  50:     monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(cfg)])
  51: 
  52:     main()
  53: 
  54:     out_dir = tmp_path / "out"
  55:     assert (out_dir / "summary.json").exists()
  56:     assert (out_dir / "sweep_results.csv").exists()
  57:     assert (out_dir / "reproducibility.json").exists()
```

## `tests/test_metrics_hardware.py`

- Total lines: **72**
- Purpose: Automated test coverage and fixtures.

```text
   1: import torch
   2: from torch import nn
   3: from torch.utils.data import DataLoader, TensorDataset
   4: 
   5: from edge_opt.hardware import estimate_layerwise_stats, summarize_hardware
   6: from edge_opt.metrics import collect_metrics
   7: from edge_opt.model import SmallCNN
   8: 
   9: 
  10: def _loader() -> DataLoader:
  11:     x = torch.randn(8, 1, 28, 28)
  12:     y = torch.randint(0, 10, (8,))
  13:     return DataLoader(TensorDataset(x, y), batch_size=4)
  14: 
  15: 
  16: def test_collect_metrics_reports_ci_and_energy_note() -> None:
  17:     model = SmallCNN()
  18:     loader = _loader()
  19:     metrics = collect_metrics(
  20:         model,
  21:         loader,
  22:         device=torch.device("cpu"),
  23:         power_watts=2.0,
  24:         precision="fp32",
  25:         benchmark_repeats=1,
  26:         benchmark_trials=2,
  27:     )
  28:     assert metrics.accuracy_ci95_low <= metrics.accuracy <= metrics.accuracy_ci95_high
  29:     assert "not measured" in metrics.energy_proxy_note
  30:     assert metrics.estimated_runtime_memory_mb > metrics.model_memory_mb
  31: 
  32: 
  33: def test_collect_metrics_rejects_empty_loader() -> None:
  34:     x = torch.randn(0, 1, 28, 28)
  35:     y = torch.randint(0, 10, (0,))
  36:     loader = DataLoader(TensorDataset(x, y), batch_size=4)
  37:     with torch.no_grad():
  38:         model = SmallCNN()
  39:         try:
  40:             collect_metrics(model, loader, torch.device("cpu"), 2.0, "fp32")
  41:             raise AssertionError("expected ValueError")
  42:         except ValueError as exc:
  43:             assert "No batches" in str(exc)
  44: 
  45: 
  46: def test_estimate_layerwise_stats_uses_dtype_sizes() -> None:
  47:     model = SmallCNN().half()
  48:     df = estimate_layerwise_stats(model, batch_size=2, activation_bytes_per_value=2)
  49:     assert int(df.loc[df["layer"] == "conv1", "parameter_bytes"].iloc[0]) % 2 == 0
  50: 
  51: 
  52: def test_estimate_layerwise_stats_supports_generic_cnn() -> None:
  53:     class TinyNet(nn.Module):
  54:         def __init__(self) -> None:
  55:             super().__init__()
  56:             self.stem = nn.Conv2d(1, 4, kernel_size=3, padding=1)
  57:             self.head = nn.Linear(4 * 28 * 28, 3)
  58: 
  59:         def forward(self, x: torch.Tensor) -> torch.Tensor:
  60:             x = self.stem(x)
  61:             return self.head(x.flatten(start_dim=1))
  62: 
  63:     df = estimate_layerwise_stats(TinyNet(), batch_size=2)
  64:     assert set(df["layer"].tolist()) == {"stem", "head"}
  65:     assert (df["macs"] > 0).all()
  66: 
  67: 
  68: def test_summarize_hardware_reports_roofline_bound() -> None:
  69:     layerwise_df = estimate_layerwise_stats(SmallCNN(), batch_size=2)
  70:     summary = summarize_hardware(layerwise_df, latency_ms=2.0, memory_bandwidth_gbps=10.0, peak_compute_gmacs=1.0)
  71:     assert "bound_regime" in summary
  72:     assert summary["bound_regime"] in {"memory-bound", "compute-bound"}
```

## `tests/test_pruning.py`

- Total lines: **17**
- Purpose: Automated test coverage and fixtures.

```text
   1: import torch
   2: 
   3: from edge_opt.model import SmallCNN
   4: from edge_opt.pruning import structured_channel_prune
   5: 
   6: 
   7: def test_structured_pruning_remaps_classifier_channels() -> None:
   8:     model = SmallCNN(conv1_channels=4, conv2_channels=6, num_classes=3)
   9:     with torch.no_grad():
  10:         model.classifier.weight.copy_(torch.arange(18, dtype=torch.float32).reshape(3, 6))
  11: 
  12:     pruned = structured_channel_prune(model, pruning_level=0.5)
  13: 
  14:     assert pruned.classifier.in_features == pruned.conv2.out_channels
  15:     # remapped classifier columns should be a subset of original channels
  16:     for row in pruned.classifier.weight:
  17:         assert torch.all(torch.isin(row, model.classifier.weight.flatten()))
```

## `tests/test_regression.py`

- Total lines: **44**
- Purpose: Automated test coverage and fixtures.

```text
   1: from __future__ import annotations
   2: 
   3: import json
   4: from pathlib import Path
   5: 
   6: import torch
   7: from torch.utils.data import DataLoader, TensorDataset
   8: 
   9: from edge_opt.experiments import run_sweep
  10: from edge_opt.model import SmallCNN, set_deterministic
  11: 
  12: 
  13: def test_minimal_regression_snapshot(tmp_path: Path) -> None:
  14:     set_deterministic(123)
  15:     x = torch.randn(8, 1, 28, 28)
  16:     y = torch.randint(0, 10, (8,))
  17:     loader = DataLoader(TensorDataset(x, y), batch_size=4)
  18: 
  19:     df = run_sweep(
  20:         base_model=SmallCNN(),
  21:         train_loader=loader,
  22:         val_loader=loader,
  23:         calibration_loader=loader,
  24:         device=torch.device("cpu"),
  25:         pruning_levels=[0.0],
  26:         precisions=["fp32"],
  27:         power_watts=2.0,
  28:         calibration_batches=1,
  29:         memory_budgets_mb=[10.0],
  30:         active_memory_budget_mb=10.0,
  31:         latency_multiplier=1.0,
  32:         benchmark_repeats=1,
  33:         benchmark_trials=1,
  34:         output_dir=tmp_path,
  35:     )
  36: 
  37:     current = {
  38:         "accuracy": float(df.iloc[0]["accuracy"]),
  39:         "model_memory_mb": float(df.iloc[0]["model_memory_mb"]),
  40:     }
  41:     baseline_path = Path("tests/baseline_metrics.json")
  42:     baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
  43:     assert abs(current["accuracy"] - baseline["accuracy"]) < 0.05
  44:     assert abs(current["model_memory_mb"] - baseline["model_memory_mb"]) < 0.05
```

