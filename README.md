# Hardware-Aware Machine Learning Pipeline for Edge Deployment

A reproducible research-oriented framework for training, compressing, and benchmarking compact neural networks under edge hardware constraints.

## Overview
This repository implements an end-to-end machine learning workflow for studying model behavior in resource-constrained environments. The system couples deterministic training and evaluation with hardware-aware analysis to quantify trade-offs among predictive performance, latency, memory footprint, and energy proxy metrics.

The project addresses a central challenge in edge AI engineering: model quality alone is insufficient when deployment targets have strict compute and memory limits. By integrating pruning, precision-aware evaluation, constraint filtering, and structured artifact export in a single configuration-driven pipeline, the repository supports repeatable experiments and transparent comparison of optimization decisions.

## Project Motivation
Modern edge deployments require models that satisfy application-level constraints (e.g., latency and memory budgets) while maintaining acceptable task performance. This repository is motivated by three research and engineering priorities:

- **Edge feasibility:** evaluate compact CNN variants for scenarios where compute and memory resources are bounded.
- **Hardware-aware optimization:** measure the impact of structured pruning and numeric precision choices on systems-oriented metrics.
- **Deterministic experimentation:** ensure that repeated runs remain comparable through fixed seeds, controlled data loading, and explicit configuration.

## System Architecture
The pipeline is organized as modular components that mirror a typical ML systems research workflow:

- **Data Pipeline**  
  Builds train/validation loaders for configured datasets and subset sizes with deterministic controls.

- **Model Training**  
  Trains a baseline compact CNN under fixed optimization settings.

- **Model Compression**  
  Applies structured channel pruning sweeps and precision variants (FP32, FP16, INT8).

- **Hardware-Aware Evaluation**  
  Computes accuracy and systems metrics, including latency, throughput, memory usage, and energy proxy; enforces memory-budget constraints; and derives Pareto frontiers.

- **Inference / Deployment Analysis**  
  Produces deployment-oriented summaries, layer-wise hardware statistics, and tabular/plot artifacts for downstream comparison.

## Repository Structure
- **`src/edge_opt/`**  
  Core implementation of configuration loading, data handling, model definition, training/evaluation orchestration, pruning/quantization, metrics, hardware analysis, deployment simulation, and reporting.

- **`scripts/`**  
  CLI entry points for running the full hardware-aware optimization pipeline.

- **`configs/`**  
  YAML experiment configurations controlling seeds, dataset subsets, pruning levels, precision modes, and hardware budget settings.

- **`docs/`**  
  Supplemental technical notes on architecture and hardware analysis.

- **`outputs/`** *(generated at runtime)*  
  Experiment artifacts such as sweep tables, frontier CSV files, summary metadata, and analysis plots.

## Features
- Deterministic, configuration-driven ML experimentation.
- Hardware-aware benchmarking with latency, throughput, memory, and energy proxy metrics.
- Structured pruning sweeps with precision-aware comparisons.
- Constraint-first evaluation via configurable memory budgets.
- Automated artifact generation for reproducible analysis.
- CLI-based pipeline execution for consistent experiment orchestration.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Usage
Run the default experiment pipeline:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

To use a custom configuration, provide a different YAML file:

```bash
python scripts/run_pipeline.py --config <path-to-config>.yaml
```

## Reproducibility
Reproducibility is supported through explicit configuration and deterministic controls:

- **Configuration files:** experiment settings are specified in YAML (e.g., dataset subsets, optimization sweeps, and hardware constraints).
- **Deterministic seeds:** global and dataloader seeds are configured to reduce run-to-run variability.
- **Experiment artifacts:** each run writes structured outputs (tables, frontiers, summaries, and plots) to enable traceable comparisons.

For strict comparability across runs, keep seed values, dataset subsets, batch size, benchmark repeat counts, and memory-budget settings fixed.

## Related Projects
This repository is part of a broader portfolio focused on hardware-aware machine learning, edge AI optimization, deterministic ML pipelines, and production ML systems.

- `neural-network-from-scratch`
- `classification-of-handwritten-digits1`
- `edge-ai-hardware-optimization`
- `data-analysis-for-hospitals`
- `nba-data-preprocessing`
- `Data-Science-AI-Portfolio`

## Future Work
Potential extensions include:

- Deployment studies on embedded and heterogeneous edge hardware.
- Additional compression strategies beyond current pruning/precision workflows.
- Expanded benchmarking methodology for improved hardware realism and cross-platform comparability.

## License
This project is released under the terms of the license provided in [`LICENSE`](LICENSE).
